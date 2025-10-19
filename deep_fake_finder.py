from keras.src.utils.module_utils import tensorflow
from PIL import Image
import numpy

filepath = "PATHTOIMAGES"

images = Image.open(filepath).convert('RGB')
images = images.resize(((256, 256)))

images = numpy.asarray(image)
images = images[None,:,:,:]


model = tensorflow.keras.models.load_model("modelfolder\\model.keras")
predict = model.predict([images])
print(predict)


