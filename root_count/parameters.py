#   ================================
#   Application and data paths
#   ================================
home_dir = "/home/pszjka/CNN/root_count/"
data_dir = "/db/pszjka/beast/db2/"

#home_dir = ""
#data_dir = ""

# =========
# Dataset
# =========
#dataset = "cassava_dataset_synthetic/old_cassava_real_synthetic_aug"
#dataset = "cassava_dataset_synthetic/young_cassava_aug"
dataset = "cassava_dataset_synthetic/old_young_cassava_aug"
#dataset = "cassava_dataset/young_cassava"
#   cassava_dataset/old_young_cassava , cassava_dataset/young_cassava, cassava_dataset/old_cassava

# ========================
# image related parameters
# ========================
channels = 3

img_cols = 256 # 720
img_rows = 256 # 960

# ============================
# Training data paths
# ============================
train = data_dir + dataset + '/train/'
trainannot = data_dir + dataset + '/trainannot/'
trainannot_rgb = data_dir + dataset + '/trainannot_rgb/'
trainannot_pred = data_dir + dataset + '/trainannot_pred/'

# ============================
# Training and Val data paths
# ============================
val = data_dir + dataset + '/val/'
valannot = data_dir + dataset + '/valannot/'
valannot_rgb = data_dir + dataset + '/valannot_rgb/'
valannot_pred = data_dir + dataset + '/valannot_pred/'

# ============================
# Training and Val data paths
# ============================
test = data_dir + dataset + '/test_real/'
testannot = data_dir + dataset + '/testannot/'
testannot_rgb = data_dir + dataset + '/testannot_rgb/'
testannot_pred = data_dir + dataset + '/testannot_pred/'

# =================================
# model training related parameters
# =================================
# unet_lite / segnet_lite / unet / segnet / densenet_121 / smallVGGnet / densenet_lite_type
model_type = "densenet_lite_type"
misc_dir = home_dir + 'misc/' + dataset + '/' + model_type
misc_dir_eval = home_dir + 'misc/' + dataset + '/' + model_type
num_epoch = 200
batch_size = 32
data_gen = False

class_weighting = [32.528058, 0.51134916]

#   8 = old_young_cassava / young_cassava
#   7 = old_cassava with 5 storage being synthetically generated

count_classes = 7
#count_classes = 8
type_classes = 2

