#   ================================
#   Application and data paths
#   ================================
home_dir = "/home/pszjka/CNN/cassava_seg/"
data_dir = "/db/pszjka/beast/db2/"

#home_dir = ""
#data_dir = ""

# =========
# Dataset
# =========
dataset = "cassava_dataset_synthetic/young_cassava_aug"

# ========================
# image related parameters
# ========================
channels = 3

img_cols = 480 # 720
img_rows = 640 # 960

# ============================
# Training and Val data paths
# ============================
trainval = data_dir + dataset + '/trainval/'
trainvalannot = data_dir + dataset + '/trainvalannot/'
trainvalannot_multi = data_dir + dataset + '/trainvalannot_multi/'

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
test = data_dir + dataset + '/test/'  # *********************
testannot = data_dir + dataset + '/testannot/'
testannot_rgb = data_dir + dataset + '/testannot_rgb/'  # *********************
testannot_pred = data_dir + dataset + '/testannot_pred/'

# ==================================
# model training related parameters
# =================================
model_type = "segnet_lite"
misc_dir = home_dir + 'misc/' + dataset + '/' + model_type
misc_dir_eval = home_dir + 'misc/' + dataset + '/' + model_type

num_epoch = 200
batch_size = 2

seg_classes = 2  #
count_classes = 8
class_weighting = [0.51134916, 32.528058]   # ******************
