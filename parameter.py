char_elements = '_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

letters = [letter for letter in char_elements]

params = {
    'char_elements' : char_elements,
    'letters' : letters,
    'num_classes' : len(letters) + 1,
    'img_w' : 128,
    'img_h' : 64,
    'batch_size' : 256,
    'val_batch_size' : 256,
    'downsample_factor' : 4,
    'max_text_len' : 12,
    'do_shuffle' : True,
    'weight_decay' : 0,
    'drop_rate' : 0.5,
    'learning_rate' : 1e-3
}
