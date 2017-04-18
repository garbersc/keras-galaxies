exec open('predict_convnet_keras_modulated.py').read() in globals()


print_weights(norm=True)
# print_weights(norm=True)


valid_scatter()
print_filters(2, norm=True)
#print_filters(3, norm=True)

save_exit()
