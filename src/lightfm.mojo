"""Compute kernels for hybrid LightFM training, inference, and scoring."""

from max.algorithm import sync_parallelize
from std.math import exp, floor, log, sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float32]()
comptime PREDICT_PARALLEL_THRESHOLD = 32_768


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def compute_representation(
    indptr: IPtr,
    indices: IPtr,
    data: FPtr,
    embeddings: FPtr,
    biases: FPtr,
    row: Int,
    components: Int,
    result: FPtr,
):
    var component = 0
    while component + W <= components:
        result.store(component, SIMD[DType.float32, W](0.0))
        component += W
    while component < components:
        result[component] = 0.0
        component += 1
    result[components] = 0.0
    var start = Int(indptr[row])
    var stop = Int(indptr[row + 1])
    for position in range(start, stop):
        var feature = Int(indices[position])
        var weight = data[position]
        component = 0
        while component + W <= components:
            var values = result.load[width=W](component)
            var feature_values = embeddings.load[width=W](
                feature * components + component
            )
            result.store(component, values + weight * feature_values)
            component += W
        while component < components:
            result[component] += (
                weight * embeddings[feature * components + component]
            )
            component += 1
        result[components] += weight * biases[feature]


def prediction(user: FPtr, item: FPtr, components: Int) -> Float32:
    var score = user[components] + item[components]
    var component = 0
    while component + W <= components:
        score += (
            user.load[width=W](component)
            * item.load[width=W](component)
        ).reduce_add()
        component += W
    while component < components:
        score += user[component] * item[component]
        component += 1
    return score


def dense_prediction(
    user_embeddings: FPtr,
    user_biases: FPtr,
    item_embeddings: FPtr,
    item_biases: FPtr,
    user: Int,
    item: Int,
    components: Int,
) -> Float32:
    var score = user_biases[user] + item_biases[item]
    var user_offset = user * components
    var item_offset = item * components
    var component = 0
    while component + W <= components:
        score += (
            user_embeddings.load[width=W](user_offset + component)
            * item_embeddings.load[width=W](item_offset + component)
        ).reduce_add()
        component += W
    while component < components:
        score += (
            user_embeddings[user_offset + component]
            * item_embeddings[item_offset + component]
        )
        component += 1
    return score


def contains(indices: IPtr, indptr: IPtr, row: Int, item: Int) -> Bool:
    var low = Int(indptr[row])
    var high = Int(indptr[row + 1])
    while low < high:
        var middle = low + (high - low) // 2
        var value = Int(indices[middle])
        if value < item:
            low = middle + 1
        else:
            high = middle
    return low < Int(indptr[row + 1]) and Int(indices[low]) == item


def next_random(state: Int) -> Int:
    return (state * 1103515245 + 12345) & 0x7fffffff


def update_value(
    values: FPtr,
    gradients: FPtr,
    momentum: FPtr,
    index: Int,
    raw_gradient: Float32,
    schedule: Int,
    learning_rate: Float32,
    rho: Float32,
    epsilon: Float32,
    alpha: Float32,
):
    var gradient = raw_gradient + alpha * values[index]
    if schedule == 1:
        gradients[index] = (
            rho * gradients[index] + (1.0 - rho) * gradient * gradient
        )
        var rate = sqrt(momentum[index] + epsilon) / sqrt(
            gradients[index] + epsilon
        )
        var change = rate * gradient
        momentum[index] = rho * momentum[index] + (1.0 - rho) * change * change
        values[index] -= change
    else:
        var rate = learning_rate / sqrt(gradients[index])
        values[index] -= rate * gradient
        gradients[index] += raw_gradient * raw_gradient


def update_pointwise(
    item_indptr: IPtr,
    item_indices: IPtr,
    item_data: FPtr,
    user_indptr: IPtr,
    user_indices: IPtr,
    user_data: FPtr,
    item_embeddings: FPtr,
    item_gradients: FPtr,
    item_momentum: FPtr,
    item_biases: FPtr,
    item_bias_gradients: FPtr,
    item_bias_momentum: FPtr,
    user_embeddings: FPtr,
    user_gradients: FPtr,
    user_momentum: FPtr,
    user_biases: FPtr,
    user_bias_gradients: FPtr,
    user_bias_momentum: FPtr,
    user_representation: FPtr,
    item_representation: FPtr,
    user: Int,
    item: Int,
    components: Int,
    loss: Float32,
    schedule: Int,
    learning_rate: Float32,
    item_alpha: Float32,
    user_alpha: Float32,
    rho: Float32,
    epsilon: Float32,
):
    var item_start = Int(item_indptr[item])
    var item_stop = Int(item_indptr[item + 1])
    var user_start = Int(user_indptr[user])
    var user_stop = Int(user_indptr[user + 1])

    for position in range(item_start, item_stop):
        var feature = Int(item_indices[position])
        var weight = item_data[position]
        update_value(
            item_biases,
            item_bias_gradients,
            item_bias_momentum,
            feature,
            loss * weight,
            schedule,
            learning_rate,
            rho,
            epsilon,
            item_alpha,
        )
        for component in range(components):
            var index = feature * components + component
            update_value(
                item_embeddings,
                item_gradients,
                item_momentum,
                index,
                loss * user_representation[component] * weight,
                schedule,
                learning_rate,
                rho,
                epsilon,
                item_alpha,
            )

    for position in range(user_start, user_stop):
        var feature = Int(user_indices[position])
        var weight = user_data[position]
        update_value(
            user_biases,
            user_bias_gradients,
            user_bias_momentum,
            feature,
            loss * weight,
            schedule,
            learning_rate,
            rho,
            epsilon,
            user_alpha,
        )
        for component in range(components):
            var index = feature * components + component
            update_value(
                user_embeddings,
                user_gradients,
                user_momentum,
                index,
                loss * item_representation[component] * weight,
                schedule,
                learning_rate,
                rho,
                epsilon,
                user_alpha,
            )


def update_pairwise(
    item_indptr: IPtr,
    item_indices: IPtr,
    item_data: FPtr,
    user_indptr: IPtr,
    user_indices: IPtr,
    user_data: FPtr,
    item_embeddings: FPtr,
    item_gradients: FPtr,
    item_momentum: FPtr,
    item_biases: FPtr,
    item_bias_gradients: FPtr,
    item_bias_momentum: FPtr,
    user_embeddings: FPtr,
    user_gradients: FPtr,
    user_momentum: FPtr,
    user_biases: FPtr,
    user_bias_gradients: FPtr,
    user_bias_momentum: FPtr,
    user_representation: FPtr,
    positive_representation: FPtr,
    negative_representation: FPtr,
    user: Int,
    positive_item: Int,
    negative_item: Int,
    components: Int,
    loss: Float32,
    schedule: Int,
    learning_rate: Float32,
    item_alpha: Float32,
    user_alpha: Float32,
    rho: Float32,
    epsilon: Float32,
):
    var positive_start = Int(item_indptr[positive_item])
    var positive_stop = Int(item_indptr[positive_item + 1])
    var negative_start = Int(item_indptr[negative_item])
    var negative_stop = Int(item_indptr[negative_item + 1])
    var user_start = Int(user_indptr[user])
    var user_stop = Int(user_indptr[user + 1])

    for position in range(positive_start, positive_stop):
        var feature = Int(item_indices[position])
        var weight = item_data[position]
        update_value(
            item_biases,
            item_bias_gradients,
            item_bias_momentum,
            feature,
            -loss * weight,
            schedule,
            learning_rate,
            rho,
            epsilon,
            item_alpha,
        )
        for component in range(components):
            var index = feature * components + component
            update_value(
                item_embeddings,
                item_gradients,
                item_momentum,
                index,
                -loss * user_representation[component] * weight,
                schedule,
                learning_rate,
                rho,
                epsilon,
                item_alpha,
            )

    for position in range(negative_start, negative_stop):
        var feature = Int(item_indices[position])
        var weight = item_data[position]
        update_value(
            item_biases,
            item_bias_gradients,
            item_bias_momentum,
            feature,
            loss * weight,
            schedule,
            learning_rate,
            rho,
            epsilon,
            item_alpha,
        )
        for component in range(components):
            var index = feature * components + component
            update_value(
                item_embeddings,
                item_gradients,
                item_momentum,
                index,
                loss * user_representation[component] * weight,
                schedule,
                learning_rate,
                rho,
                epsilon,
                item_alpha,
            )

    for position in range(user_start, user_stop):
        var feature = Int(user_indices[position])
        var weight = user_data[position]
        update_value(
            user_biases,
            user_bias_gradients,
            user_bias_momentum,
            feature,
            loss * weight,
            schedule,
            learning_rate,
            rho,
            epsilon,
            user_alpha,
        )
        for component in range(components):
            var index = feature * components + component
            var item_difference = (
                negative_representation[component]
                - positive_representation[component]
            )
            update_value(
                user_embeddings,
                user_gradients,
                user_momentum,
                index,
                loss * item_difference * weight,
                schedule,
                learning_rate,
                rho,
                epsilon,
                user_alpha,
            )


@export("mlfm_representations")
def mlfm_representations(
    indptr_addr: Int,
    indices_addr: Int,
    data_addr: Int,
    embeddings_addr: Int,
    biases_addr: Int,
    rows: Int,
    components: Int,
    result_embeddings_addr: Int,
    result_biases_addr: Int,
) abi("C"):
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var data = fp(data_addr)
    var embeddings = fp(embeddings_addr)
    var biases = fp(biases_addr)
    var result_embeddings = fp(result_embeddings_addr)
    var result_biases = fp(result_biases_addr)
    for row in range(rows):
        var row_offset = row * components
        var component = 0
        while component + W <= components:
            result_embeddings.store(
                row_offset + component, SIMD[DType.float32, W](0.0)
            )
            component += W
        while component < components:
            result_embeddings[row_offset + component] = 0.0
            component += 1
        result_biases[row] = 0.0
        var start = Int(indptr[row])
        var stop = Int(indptr[row + 1])
        for position in range(start, stop):
            var feature = Int(indices[position])
            var weight = data[position]
            component = 0
            while component + W <= components:
                var values = result_embeddings.load[width=W](
                    row_offset + component
                )
                var feature_values = embeddings.load[width=W](
                    feature * components + component
                )
                result_embeddings.store(
                    row_offset + component, values + weight * feature_values
                )
                component += W
            while component < components:
                result_embeddings[row_offset + component] += (
                    weight * embeddings[feature * components + component]
                )
                component += 1
            result_biases[row] += weight * biases[feature]


@export("mlfm_predict_dense")
def mlfm_predict_dense(
    user_embeddings_addr: Int,
    user_biases_addr: Int,
    item_embeddings_addr: Int,
    item_biases_addr: Int,
    user_ids_addr: Int,
    item_ids_addr: Int,
    pairs: Int,
    components: Int,
    result_addr: Int,
    workers_arg: Int,
) abi("C"):
    var user_embeddings = fp(user_embeddings_addr)
    var user_biases = fp(user_biases_addr)
    var item_embeddings = fp(item_embeddings_addr)
    var item_biases = fp(item_biases_addr)
    var user_ids = ip(user_ids_addr)
    var item_ids = ip(item_ids_addr)
    var result = fp(result_addr)
    var workers = min(max(workers_arg, 1), pairs)

    @parameter
    def predict_chunk(worker: Int):
        var start = worker * pairs // workers
        var stop = (worker + 1) * pairs // workers
        for pair in range(start, stop):
            result[pair] = dense_prediction(
                user_embeddings,
                user_biases,
                item_embeddings,
                item_biases,
                Int(user_ids[pair]),
                Int(item_ids[pair]),
                components,
            )

    if workers > 1 and pairs >= PREDICT_PARALLEL_THRESHOLD:
        sync_parallelize[predict_chunk](workers)
    else:
        for pair in range(pairs):
            result[pair] = dense_prediction(
                user_embeddings,
                user_biases,
                item_embeddings,
                item_biases,
                Int(user_ids[pair]),
                Int(item_ids[pair]),
                components,
            )


@export("mlfm_predict")
def mlfm_predict(
    item_indptr_addr: Int,
    item_indices_addr: Int,
    item_data_addr: Int,
    user_indptr_addr: Int,
    user_indices_addr: Int,
    user_data_addr: Int,
    item_embeddings_addr: Int,
    item_biases_addr: Int,
    user_embeddings_addr: Int,
    user_biases_addr: Int,
    user_ids_addr: Int,
    item_ids_addr: Int,
    pairs: Int,
    components: Int,
    result_addr: Int,
    user_work_addr: Int,
    item_work_addr: Int,
) abi("C"):
    var item_indptr = ip(item_indptr_addr)
    var item_indices = ip(item_indices_addr)
    var item_data = fp(item_data_addr)
    var user_indptr = ip(user_indptr_addr)
    var user_indices = ip(user_indices_addr)
    var user_data = fp(user_data_addr)
    var item_embeddings = fp(item_embeddings_addr)
    var item_biases = fp(item_biases_addr)
    var user_embeddings = fp(user_embeddings_addr)
    var user_biases = fp(user_biases_addr)
    var user_ids = ip(user_ids_addr)
    var item_ids = ip(item_ids_addr)
    var result = fp(result_addr)
    var user_work = fp(user_work_addr)
    var item_work = fp(item_work_addr)
    for pair in range(pairs):
        compute_representation(
            user_indptr,
            user_indices,
            user_data,
            user_embeddings,
            user_biases,
            Int(user_ids[pair]),
            components,
            user_work,
        )
        compute_representation(
            item_indptr,
            item_indices,
            item_data,
            item_embeddings,
            item_biases,
            Int(item_ids[pair]),
            components,
            item_work,
        )
        result[pair] = prediction(user_work, item_work, components)


@export("mlfm_score_matrix")
def mlfm_score_matrix(
    user_embeddings_addr: Int,
    user_biases_addr: Int,
    item_embeddings_addr: Int,
    item_biases_addr: Int,
    user_ids_addr: Int,
    users: Int,
    items: Int,
    components: Int,
    result_addr: Int,
) abi("C"):
    var user_embeddings = fp(user_embeddings_addr)
    var user_biases = fp(user_biases_addr)
    var item_embeddings = fp(item_embeddings_addr)
    var item_biases = fp(item_biases_addr)
    var user_ids = ip(user_ids_addr)
    var result = fp(result_addr)
    for user_position in range(users):
        var user = Int(user_ids[user_position])
        for item in range(items):
            result[user_position * items + item] = dense_prediction(
                user_embeddings,
                user_biases,
                item_embeddings,
                item_biases,
                user,
                item,
                components,
            )


@export("mlfm_fit_epoch")
def mlfm_fit_epoch(
    loss_kind: Int,
    schedule: Int,
    item_indptr_addr: Int,
    item_indices_addr: Int,
    item_data_addr: Int,
    user_indptr_addr: Int,
    user_indices_addr: Int,
    user_data_addr: Int,
    positive_indptr_addr: Int,
    positive_indices_addr: Int,
    interaction_users_addr: Int,
    interaction_items_addr: Int,
    interaction_values_addr: Int,
    sample_weights_addr: Int,
    order_addr: Int,
    examples: Int,
    item_embeddings_addr: Int,
    item_gradients_addr: Int,
    item_momentum_addr: Int,
    item_biases_addr: Int,
    item_bias_gradients_addr: Int,
    item_bias_momentum_addr: Int,
    user_embeddings_addr: Int,
    user_gradients_addr: Int,
    user_momentum_addr: Int,
    user_biases_addr: Int,
    user_bias_gradients_addr: Int,
    user_bias_momentum_addr: Int,
    items: Int,
    components: Int,
    learning_rate_arg: Float64,
    item_alpha_arg: Float64,
    user_alpha_arg: Float64,
    rho_arg: Float64,
    epsilon_arg: Float64,
    max_sampled: Int,
    random_state_arg: Int,
    user_work_addr: Int,
    positive_work_addr: Int,
    negative_work_addr: Int,
) abi("C") -> Int:
    var item_indptr = ip(item_indptr_addr)
    var item_indices = ip(item_indices_addr)
    var item_data = fp(item_data_addr)
    var user_indptr = ip(user_indptr_addr)
    var user_indices = ip(user_indices_addr)
    var user_data = fp(user_data_addr)
    var positive_indptr = ip(positive_indptr_addr)
    var positive_indices = ip(positive_indices_addr)
    var interaction_users = ip(interaction_users_addr)
    var interaction_items = ip(interaction_items_addr)
    var interaction_values = fp(interaction_values_addr)
    var sample_weights = fp(sample_weights_addr)
    var order = ip(order_addr)
    var item_embeddings = fp(item_embeddings_addr)
    var item_gradients = fp(item_gradients_addr)
    var item_momentum = fp(item_momentum_addr)
    var item_biases = fp(item_biases_addr)
    var item_bias_gradients = fp(item_bias_gradients_addr)
    var item_bias_momentum = fp(item_bias_momentum_addr)
    var user_embeddings = fp(user_embeddings_addr)
    var user_gradients = fp(user_gradients_addr)
    var user_momentum = fp(user_momentum_addr)
    var user_biases = fp(user_biases_addr)
    var user_bias_gradients = fp(user_bias_gradients_addr)
    var user_bias_momentum = fp(user_bias_momentum_addr)
    var user_work = fp(user_work_addr)
    var positive_work = fp(positive_work_addr)
    var negative_work = fp(negative_work_addr)
    var learning_rate = Float32(learning_rate_arg)
    var item_alpha = Float32(item_alpha_arg)
    var user_alpha = Float32(user_alpha_arg)
    var rho = Float32(rho_arg)
    var epsilon = Float32(epsilon_arg)
    var random_state = random_state_arg

    for example_position in range(examples):
        var example = Int(order[example_position])
        var interaction = interaction_values[example]
        if loss_kind != 0 and interaction <= 0.0:
            continue
        var user = Int(interaction_users[example])
        var positive_item = Int(interaction_items[example])
        compute_representation(
            user_indptr,
            user_indices,
            user_data,
            user_embeddings,
            user_biases,
            user,
            components,
            user_work,
        )
        compute_representation(
            item_indptr,
            item_indices,
            item_data,
            item_embeddings,
            item_biases,
            positive_item,
            components,
            positive_work,
        )

        if loss_kind == 0:
            var score = prediction(user_work, positive_work, components)
            var probability = 1.0 / (1.0 + exp(-score))
            var target = Float32(1.0) if interaction > 0.0 else Float32(0.0)
            var point_loss = sample_weights[example] * (probability - target)
            update_pointwise(
                item_indptr,
                item_indices,
                item_data,
                user_indptr,
                user_indices,
                user_data,
                item_embeddings,
                item_gradients,
                item_momentum,
                item_biases,
                item_bias_gradients,
                item_bias_momentum,
                user_embeddings,
                user_gradients,
                user_momentum,
                user_biases,
                user_bias_gradients,
                user_bias_momentum,
                user_work,
                positive_work,
                user,
                positive_item,
                components,
                point_loss,
                schedule,
                learning_rate,
                item_alpha,
                user_alpha,
                rho,
                epsilon,
            )
            continue

        var negative_item = 0
        var sampled = 0
        var found = False
        var positive_score = prediction(user_work, positive_work, components)
        var pair_loss = Float32(0.0)
        var sampling_limit = examples if loss_kind == 1 else max_sampled
        while sampled < sampling_limit:
            random_state = next_random(random_state)
            if loss_kind == 1:
                negative_item = Int(interaction_items[random_state % examples])
            else:
                negative_item = random_state % items
            sampled += 1
            if contains(positive_indices, positive_indptr, user, negative_item):
                continue
            compute_representation(
                item_indptr,
                item_indices,
                item_data,
                item_embeddings,
                item_biases,
                negative_item,
                components,
                negative_work,
            )
            var negative_score = prediction(user_work, negative_work, components)
            if loss_kind == 1:
                pair_loss = sample_weights[example] * (
                    1.0
                    - 1.0
                    / (1.0 + exp(-(positive_score - negative_score)))
                )
                found = True
                break
            if negative_score > positive_score - 1.0:
                var rank_weight = log(
                    max(
                        Float64(1.0),
                        floor(Float64(items - 1) / Float64(sampled)),
                    )
                )
                pair_loss = sample_weights[example] * Float32(rank_weight)
                if pair_loss > 10.0:
                    pair_loss = 10.0
                found = True
                break
        if found:
            update_pairwise(
                item_indptr,
                item_indices,
                item_data,
                user_indptr,
                user_indices,
                user_data,
                item_embeddings,
                item_gradients,
                item_momentum,
                item_biases,
                item_bias_gradients,
                item_bias_momentum,
                user_embeddings,
                user_gradients,
                user_momentum,
                user_biases,
                user_bias_gradients,
                user_bias_momentum,
                user_work,
                positive_work,
                negative_work,
                user,
                positive_item,
                negative_item,
                components,
                pair_loss,
                schedule,
                learning_rate,
                item_alpha,
                user_alpha,
                rho,
                epsilon,
            )
    return random_state
