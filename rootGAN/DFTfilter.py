'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('150_100_fake_A_0.png')
img = img[:,:,0]
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('150_100_fake_A_0.png')

dst = cv2.fastNlMeansDenoisingColored(img,None,2,2,27,21)

cv2.imwrite("filtered.png", dst)