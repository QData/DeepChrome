import kipoi
import numpy as np

example_input = "./downloaded/example_files/input_file"
example_expr = "./downloaded/example_files/expr_file"

#DataLoader = kipoi.get_dataloader_factory(".", source="dir")
#print(DataLoader(example_input, example_expr).__getitem__(1))

model = kipoi.get_model(".", source="dir")
prediction = model.pipeline.predict({"input_file": example_input, "expr_file": example_expr}) 

print("Output Return Type: ", type(prediction))
print(len(prediction))
print(len(prediction[0]))
print(len(prediction[0][1]))
print(len(prediction[1][1]))
print(len(prediction[2][1]))
#for i in range(len(prediction)):
#	for j in range(len(prediction[0])):
#		print(len(prediction[i][j]))

#print("Output Shape: ", np.array(prediction).shape)