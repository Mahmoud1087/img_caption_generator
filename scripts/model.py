from src import data_generator, load_testing_imgs, generate_captions_for_images
from tensorflow.keras.models import load_model


def train_and_test(epochs:int, descriptions:dict, features:dict, tokenizer, max_length:int, vocab_size:int):
    
    EPOCHS = epochs
    STEPS = len(descriptions)

    for i in range(EPOCHS):
        
        model.fit(data_generator(descriptions=descriptions,
                                features=image_feature_vectors,
                                tokenizer=tk,
                                max_length=max_caps,
                                vocab_size=vocab_size),
                epochs=1, 
                steps_per_epoch=STEPS, 
                verbose=1)
        
        model.save("../models/baseline_model/baseline_model_" + str(i) + ".keras")
        

    model = load_model("../models/baseline_model/baseline_model_4.keras")

    features = load_testing_imgs(directory="../data/raw/Flickr8k_Dataset/Flicker8k_Dataset",
                                txt_file="../data/raw/Flickr8k_text/Flickr_8k.devImages.txt")

    caps = generate_captions_for_images(model=model, 
                                        tokenizer=tk, 
                                        features=features, 
                                        max_length=max_caps)


# Base model
def model(epochs:int, descriptions:dict, features:dict, tokenizer, max_length:int, vocab_size:int):
    
    # CNN model inputs
    img_inputs = Input(shape=(2048,))
    fe1 = Dropout(0.5)(img_inputs)
    fe2 = Dense(256)(fe1)

    # LSTM model inputs
    seq_inputs = Input(shape=(max_caps,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(seq_inputs)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Adding the models
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[img_inputs, seq_inputs], 
                outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                optimizer='Adam')

    train_and_test(epochs=epochs, 
                   descriptions=descriptions, 
                   features=features, 
                   tokenizer=tokenizer, 
                   max_length=max_length, 
                   vocab_size=vocab_size)


if __name__ == "__main__":
    
    model()
