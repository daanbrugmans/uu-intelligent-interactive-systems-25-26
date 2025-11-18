from datasets import load_dataset
import sklearn.model_selection as skl_ms
import sklearn.discriminant_analysis as skl_da

# dataset = load_dataset("imagefolder", data_dir = "./DiffusionFER/DiffusionEmotion_S/Cropped")
# classes = set(dataset["cropped"]["label"])

# landmark_features = []
# skipped = 0
# skipped_indices = []
# # for index, img in enumerate(dataset)

# x_train, y_train, x_test, y_test = skl_ms.train_test_split(dataset, labels, test_size = 0.3)

# # training an LDA
# lda = skl_da.LinearDiscriminantAnalysis()
# lda.fit(x_train, y_train)