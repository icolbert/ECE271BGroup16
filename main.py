from matplotlib import pyplot as plt

from utils import *

if __name__ == '__main__':
	img = cv2.imread('/Users/josejoy/Desktop/ECE 271B Stat Learning /project/eq1.jpg',0)
	segments= extract_segments(img, 10)
	plt.imshow(segments[1])
	plt.show()