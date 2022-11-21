# Show Result
ix = random.randint(0, len(Inputs))

img = Inputs[ix]
mask = Ground_Truth[ix]
resized = Image_Resize[ix]
gray = Gray_Scale[ix]

print("Input Image")
imshow(img)
plt.show()

print('Mask')
imshow(mask[:, :, 0])
plt.show()

print("Resized Image")
imshow(resized)
plt.show()

print('GrayScale Image')
imshow(gray)
plt.show()