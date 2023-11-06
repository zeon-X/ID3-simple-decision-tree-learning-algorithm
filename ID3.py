import math


def calculate_whole_entropy(data, feature_index):
    total_samples = len(data)
    feature_entropy = 0.0

    unique_values = set(data[i][feature_index] for i in range(total_samples))

    # print("unique_values", unique_values)

    for value in unique_values:
        subset_data = [data[i] for i in range(
            total_samples) if data[i][feature_index] == value]

        subset_size = len(subset_data)
        # print(subset_size)

        if subset_size == 0:
            continue

        class_counts = [row[-1] for row in subset_data]

        # class_probabilities = [class_counts.count(
        #     c) / subset_size for c in set(class_counts)]

        class_probabilities = len(class_counts) / total_samples

        subset_entropy = -class_probabilities * math.log2(class_probabilities)

        # print(class_probabilities)
        # print(subset_entropy)

        feature_entropy += subset_entropy

    return feature_entropy


def calculate_feature_entropy(data, feature_index):
    total_samples = len(data)
    feature_entropy = 0.0
    subset_entropies = []  # Create an array to store subset entropies

    unique_values = set(data[i][feature_index] for i in range(total_samples))

    # print("unique_values", unique_values)

    for value in unique_values:
        subset_data = [data[i] for i in range(
            total_samples) if data[i][feature_index] == value]

        subset_size = len(subset_data)

        if subset_size == 0:
            continue

        class_counts = [row[-1] for row in subset_data]
        class_probabilities = [class_counts.count(
            c) / subset_size for c in set(class_counts)]

        subset_entropy = 0.0

        for p in class_probabilities:
            if p > 0:
                subset_entropy -= p * math.log2(p)

        feature_entropy += (subset_size / total_samples) * subset_entropy

        # Store (value, subset_entropy) tuple
        subset_entropies.append((value, subset_entropy))

        # print("value:", value)
        # print("class_probabilities:", class_probabilities)
        # print("subset_size:", subset_size)
        # print("subset_entropy:", subset_entropy)
        # print("subset_entropies:", subset_entropies)
        # print("entropy (probability * Ent):", (subset_size / total_samples), " * ", subset_entropy, " = ",
        #       (subset_size / total_samples) * subset_entropy)

    # Return both the feature entropy and subset entropies
    # print("feature_entropy:", feature_entropy)
    # print("\n")
    return feature_entropy, subset_entropies


def calculate_information_gain(data):
    target_entropy = calculate_whole_entropy(data, feature_index=-1)
    num_features = len(data[0]) - 1  # Exclude the target feature
    information_gains = []

    for feature_index in range(num_features):
        feature_entropy, subset_entropies = calculate_feature_entropy(
            data, feature_index)
        information_gain = target_entropy - feature_entropy
        information_gains.append((information_gain, subset_entropies))

    return information_gains


def get_me_max_gain(information_gains):
    maxGain = (0, (0, []))
    for i, ig in enumerate(information_gains):
        # print(maxGain[1])
        if maxGain[1][0] < ig[0]:
            maxGain = (i, ig)

    return maxGain

# -----------Example usage with your provided dataset
# data = [
#     [2, 3, 1, 1, 1],  # 36-55, Master's, High, Single, Yes
#     [1, 1, 0, 1, 0],  # 18-35, High School, Low, Single, No
#     [2, 2, 0, 1, 1],  # 36-55, Bachelor's, Low, Single, Yes
#     [1, 0, 1, 1, 0],  # 18-35, Bachelor's, High, Single, No
#     [0, 1, 0, 1, 1],  # <18, High School, Low, Single, Yes
#     [1, 0, 1, 0, 0],  # 18-35, Bachelor's, High, Married, No
#     [2, 1, 0, 0, 0],  # 36-55, Bachelor's, Low, Married, No
#     [3, 0, 1, 1, 1],  # >55, Bachelor's, High, Single, Yes
#     [2, 3, 0, 0, 0],  # 36-55, Master's, Low, Married, No
#     [3, 3, 0, 0, 1],  # >55, Master's, Low, Married, Yes
#     [2, 3, 1, 1, 1],  # 36-55, Master's, High, Single, Yes
#     [3, 3, 1, 1, 1],  # >55, Master's, High, Single, Yes
#     [0, 1, 1, 1, 0],  # <18, High School, High, Single, No
#     [2, 3, 0, 1, 1],  # 36-55, Master's, Low, Single, Yes
#     [2, 1, 0, 1, 1],  # 36-55, High School, Low, Single, Yes
#     [0, 1, 0, 0, 1],  # <18, High School, Low, Married, Yes
#     [1, 0, 1, 0, 0],  # 18-35, Bachelor's, High, Married, No
#     [3, 1, 1, 0, 1],  # >55, High School, High, Married, Yes
#     [3, 2, 0, 1, 1]   # 36-55, High School, High, Married, No
# ]


data = [
    ["36-55", "Master's", "High", "Single", 1],
    ["18-35", "High School", "Low", "Single", 0],
    ["36-55", "Master's", "Low", "Single", 1],
    ["18-35", "Bachelor's", "High", "Single", 0],
    ["<18", "High School", "Low", "Single", 1],
    ["18-35", "Bachelor's", "High", "Married", 0],
    ["36-55", "Bachelor's", "Low", "Married", 0],
    [">55", "Bachelor's", "High", "Single", 1],
    ["36-55", "Master's", "Low", "Married", 0],
    [">55", "Master's", "Low", "Married", 1],
    ["36-55", "Master's", "High", "Single", 1],
    [">55", "Master's", "High", "Single", 1],
    ["<18", "High School", "High", "Single", 0],
    ["36-55", "Master's", "Low", "Single", 1],
    ["36-55", "High School", "Low", "Single", 1],
    ["<18", "High School", "Low", "Married", 1],
    ["18-35", "Bachelor's", "High", "Married", 0],
    [">55", "High School", "High", "Married", 1],
    [">55", "Bachelor's", "Low", "Single", 1],
    ["36-55", "High School", "High", "Married", 0]
]
# -----------Calculate the entropy of the 'buys computer' feature (last column)

# buys_computer_entropy = calculate_whole_entropy(data, feature_index=-1)
# print("Entropy of the 'buys computer' feature:", buys_computer_entropy)

# -----------Modify this to the index of the feature you want to calculate entropy for

# feature_index = 3
# feature_entropy, subset_entropies = calculate_feature_entropy(
#     data, feature_index)
# print("Entropy of the specified feature:", feature_entropy)
# print("Subset entropies:", subset_entropies)
# print("IG:", buys_computer_entropy - feature_entropy)


def get_me_vertex(data):
    information_gains = calculate_information_gain(data)

    # Print the information gains for each feature
    # for i, ig in enumerate(information_gains):
    #     print(f"Information Gain for feature {i}: {ig}")

    maxGain = get_me_max_gain(information_gains=information_gains)
    # returns (vertex feature number, (gain, [subset_entropies]))
    print(maxGain)


get_me_vertex(data)
