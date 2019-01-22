#   ================================
#   Application and data paths
#   ================================
home_dir = "/home/pszjka/CNN/rootGAN/"
data_dir = "/db/pszjka/beast/db2/"
#home_dir = ""
#data_dir = ""

# =========
# Dataset
# =========
dataset = "cassava_gan/old_cassava_new"   # cassava_synthetic_real  / cassava_synthetic / cassava_gan / cassava_real

# ========================
# image related parameters
# ========================
channels = 3
img_cols = 720
img_rows = 960

# ============================
# Training and Val data paths
# ============================
trainval = data_dir + dataset + '/trainval/'
trainvalannot = data_dir + dataset + '/trainvalannot/'

# ===============
# Test data paths
# ===============
test_data = data_dir + dataset + '/test/'
test_data_annot = data_dir + dataset + '/testannot/'

# ============================
# Training data paths
# for classification only
# ============================
train = data_dir + dataset + '/train/'
trainannot = data_dir + dataset + '/trainannot/'

# ============================
# Training and Val data paths
# for classification only
# ============================
val = data_dir + dataset + '/val/'
valannot = data_dir + dataset + '/valannot/'


# ==================================
# model training related parameters
# =================================
# smoothGAN / texturedGAN / xception****
model_type = "texturedGAN3"
misc_dir = home_dir + 'misc/' + dataset + '/' + model_type
misc_dir2 = home_dir + 'misc/' + dataset
misc_dir_eval = home_dir + 'misc/' + dataset + '/evaluate/' + model_type
num_epoch = 1200
batch_size = 2
num_classes = 7
