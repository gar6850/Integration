import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import Input
from Encoder import Encoder
from Decoder import Decoder
class Prediction :
    embedding_dim = 256 
    units = 512
    vocab_size = 5001  #top 5,000 words +1
    max_length = 31
    feature_shape = 2048
    attention_feature_shape = 64
    

    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.initialize()

    def initialize(self):
       
        all_img_problems = [] 
        top_word_cnt = 5000
        
        with open('/cluster/home/guillera/mode_3_medical/isu-chest-data/archive/ireport.txt' , 'r') as fo:
            next(fo) 
            contador = 0
            for line in fo :
                split_arr = line.split(',')
                all_img_problems.append(split_arr[2])
        annotations = ['<start>' + ' ' + line + ' ' + '<end>' for line in all_img_problems]
        self.tokenizer = Tokenizer(num_words = top_word_cnt+1, filters= '!"#$%^&*()_+.,:;-?/~`{}[]|\=@ ',
                            lower = True, char_level = False, 
                            oov_token = 'UNK')
        self.tokenizer.fit_on_texts(annotations)   


    def load_images(self,image_path) :
        IMAGE_SHAPE = (299, 299)
        img = tf.io.read_file(image_path, name = None)
        img = tf.image.decode_jpeg(img, channels=0)
        img = tf.image.resize(img, IMAGE_SHAPE)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path


    def evaluate(self, image):
        decoder=Decoder(self.embedding_dim, self.units, self.vocab_size)
        encoder=Encoder(self.embedding_dim)
        saved_decoder =  keras.models.load_model('/cluster/home/guillera/mode_3_medical/decoder_model2', compile=False)
        saved_encoder = keras.models.load_model('/cluster/home/guillera/mode_3_medical/encoder_model2', compile=False)


        image_model =  tf.keras.applications.xception.Xception(weights='imagenet', include_top = False)
        input_tensor = Input(shape=(299,299,1))
        x = tf.keras.layers.Conv2D(3,(3,3),padding='same')(input_tensor)    # x has a dimension of (img_height,img_width,3)
        out1 = image_model(x)
        #out = tf.keras.layers.Dense(10, activation = 'softmax')(out1)

        image_features_extract_model = tf.compat.v1.keras.Model(inputs = input_tensor, outputs = out1)
        #attention_plot = np.zeros((self.max_length, self.attention_feature_shape))
        image_features_extract_model = keras.models.load_model('/cluster/home/guillera/mode_3_medical/features', compile=False)


        hidden = decoder.init_state(batch_size=1)

        temp_input = tf.expand_dims(self.load_images(image)[0], 0) 
        img_tensor_val = image_features_extract_model(temp_input) 
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = saved_encoder (img_tensor_val) 

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = saved_decoder(dec_input, features, hidden) 
            #attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append (self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result,predictions

            dec_input = tf.expand_dims([predicted_id], 0)

        #attention_plot = attention_plot[:len(result), :]
        return result,predictions

    def predict_caption(self) :
        result, pred_test = self.evaluate(self.imagePath)
        pred_caption=' '.join(result).rsplit(' ', 1)[0]
        print('Prediction Caption:', pred_caption)
        return pred_caption
