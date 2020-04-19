'''
* Yapay Sinir Ağının oluşturulan örnek CAPTCHA'lar ile eğitilerek daha sonra benzer bir CAPTCHA'nın tanınmasında kullanılması.
* Yararlanılan Kaynak: Learning Data Mining with Python by Rober Layton (Chapter 8)
* Hazırlayan: Bilishim Siber Güvenlik ve Yapay Zeka tarafından söz konusu çalışma güncellenmiştir
'''


import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from matplotlib import pyplot as plt


#İstenilen boyutta bir CAPTCHA yarat
def create_captcha(text, shear=0, size=(100,28)):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
	
	#https://fontlibrary.org/en/font/bretan
    font = ImageFont.truetype(r"Coval-Regular.otf", 22)
    draw.text((2, 2), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()
	
	

'''
#Örnek bir CAPTCHA yarat
image = create_captcha("TURK", shear=0.5)
plt.imshow(image, cmap="gray")
plt.show()
'''



from skimage.measure import label, regionprops

#Oluşturulan CAPTCHA'yı harflerine bölümlendir
def segment_image(image):
    labeled_image = label(image > 0)
    subimages = []
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        return [image,]
    return subimages
	

	
'''	
#Harflerine bölümlendirilmiş CAPTCHA'yı göster
subimages = segment_image(image)
f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
for i in range(len(subimages)):
    axes[i].imshow(subimages[i], cmap="gray")
    plt.show()
'''

from sklearn.utils import check_random_state
random_state = check_random_state(14)
letters = list("ACBDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)

#İngilizce Alfabe'den seçilen bir karakter ile CAPTCHA oluştur
def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    return create_captcha(letter, shear=shear, size=(20, 20)), letters.index(letter)
	

'''	
#Rasgele üretilen bir örneği göster
image, target = generate_sample(random_state)
plt.imshow(image, cmap="gray")
plt.show()
print("The target for this image is: {0}".format(target))
'''



#3000 adet örnek CAPTCHA üret
dataset, targets = zip(*(generate_sample(random_state) for i in
range(3000)))
dataset = np.array(dataset, dtype='float')
targets = np.array(targets)


#İngilizce alfabedeki harflere karşılık gelecek tekil matrisi oluştur
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0],1))

y = y.todense()

from skimage.transform import resize
dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for
sample in dataset])

X = dataset.reshape((dataset.shape[0], dataset.shape[1] *
dataset.shape[2]))

#Yüzde seksenlik bir eğitim seti oluştur, geri kalan yüzde 20 test için kullanılacaktır
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X, y, train_size=0.8)



#https://github.com/pybrain/pybrain
#python setup.py install

from pybrain.datasets import SupervisedDataSet

training = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_train.shape[0]):
    training.addSample(X_train[i], y_train[i])
	
	
testing = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_test.shape[0]):
    testing.addSample(X_test[i], y_test[i])
	
	
from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)


from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, training, learningrate=0.01,
weightdecay=0.01)

#50 kuşaklık bir eğitim sürecini başlat
trainer.trainEpochs(epochs=100)

predictions = trainer.testOnClassData(dataset=testing)

from sklearn.metrics import f1_score
print("F-score: {0:.2f}".format(f1_score(predictions, y_test.argmax(axis=1), average='micro')))

from sklearn.metrics import classification_report
print(classification_report(y_test.argmax(axis=1), predictions))



#Paramtre olarak girilen bir CAPTCHA'nın ne olduğunu çözümle
def predict_captcha(captcha_image, neural_network):
    subimages = segment_image(captcha_image)
    predicted_word = ""
    for subimage in subimages:
        subimage = resize(subimage, (20, 20))
        outputs = net.activate(subimage.flatten())
        prediction = np.argmax(outputs)
        predicted_word += letters[prediction]
    return predicted_word
	
	
word = "TURK"
new_captcha = create_captcha(word, shear=0.2)
plt.imshow(new_captcha, cmap="gray")
print("ÇÖZÜMLENEN CAPTCHA")
print(predict_captcha(new_captcha, net))
plt.show()


		
		


