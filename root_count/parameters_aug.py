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
dataset = "cassava_dataset/old_cassava"
#   cassava_dataset/old_young_cassava , cassava_dataset/young_cassava, cassava_dataset/old_cassava

# ========================
# image related parameters
# ========================
channels = 3

img_cols = 720 # 720
img_rows = 960 # 960

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
test = data_dir + dataset + '/test/'
testannot = data_dir + dataset + '/testannot/'
testannot_rgb = data_dir + dataset + '/testannot_rgb/'
testannot_pred = data_dir + dataset + '/testannot_pred/'

# ==================================
# model training related parameters
# =================================
# unet_lite / segnet_lite / unet / segnet / densenet_121 / smallVGGnet
model_type = "densenet_121"
misc_dir = home_dir + 'misc/' + dataset + '/' + model_type
misc_dir_eval = home_dir + 'misc/' + dataset + '/' + model_type
num_epoch = 300
batch_size = 1
data_gen = False

#   8 = old_young_cassava / young_cassava
#   6 = old_cassava

count_classes = 6

