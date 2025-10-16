# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV data
# csv_file = "/home/ashwin/Project/FunieGAN/FUnIE-GAN/PyTorch/backup/FunieGAN/EUVP/training_log.csv"  # Replace with your actual CSV file name
# data = pd.read_csv(csv_file)

# # Extract relevant columns
# epochs = data["Epoch"]
# batches = data["Batch"]
# d_loss = data["D_loss"]
# g_loss = data["G_loss"]
# adv_loss = data["Adv_loss"]

# # Combine epoch and batch for x-axis
# x_axis = epochs + batches / max(batches)

# # Plot the losses as connecting lines
# plt.figure(figsize=(10, 6))
# plt.plot(x_axis, d_loss, label="D Loss", color="blue", linestyle="-", linewidth=1.5)
# plt.plot(x_axis, g_loss, label="G Loss", color="green", linestyle="-", linewidth=1.5)
# plt.plot(x_axis, adv_loss, label="Adv Loss", color="red", linestyle="-", linewidth=1.5)

# # Title and labels
# plt.title("Losses over Epochs and Batches")
# plt.xlabel("Epoch + Batch Proportion")
# plt.ylabel("Loss Value")
# plt.ylim(0, 5)  # Set the y-limit to better represent the low values
# plt.grid(True)

# # Add legend
# plt.legend()

# # Save the plot to the current directory
# output_file = "loss_plot.png"
# plt.savefig(output_file)

# # Show the plot (optional)
# plt.show()

# print(f"Plot saved as {output_file}")


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
csv_file = "/home/ashwin/Project/FunieGAN/FUnIE-GAN/PyTorch/backup/FunieGAN/EUVP/training_log.csv"  # Replace with your actual CSV file name
data = pd.read_csv(csv_file)

# Filter the data to include only rows where Batch is 5750
filtered_data = data[data["Batch"] == 5750]

# Extract relevant columns from the filtered data
epochs = filtered_data["Epoch"]
batches = filtered_data["Batch"]
d_loss = filtered_data["D_loss"]
g_loss = filtered_data["G_loss"]
adv_loss = filtered_data["Adv_loss"]

# Combine epoch and batch for x-axis (even though batch is always 5750, this will help show it proportionally)
x_axis = epochs + batches / max(batches)

# Plot the losses as connecting lines
plt.figure(figsize=(10, 6))
plt.plot(x_axis, d_loss, label="D Loss", color="blue", linestyle="-", linewidth=1.5)
plt.plot(x_axis, g_loss, label="G Loss", color="green", linestyle="-", linewidth=1.5)
plt.plot(x_axis, adv_loss, label="Adv Loss", color="red", linestyle="-", linewidth=1.5)

# Title and labels
plt.title("Losses over Epochs at Batch 5750")
plt.xlabel("Epoch + Batch Proportion")
plt.ylabel("Loss Value")
plt.ylim(0, 5)  # Set the y-limit to better represent the low values
plt.grid(True)

# Add legend
plt.legend()

# Save the plot to the current directory
output_file = "loss_plot_filtered_batch_5750.png"
plt.savefig(output_file)

# Show the plot (optional)
plt.show()

print(f"Plot saved as {output_file}")

# Find the epoch with the lowest G_loss
min_g_loss_epoch = filtered_data.loc[filtered_data["G_loss"].idxmin(), "Epoch"]
min_g_loss_value = filtered_data["G_loss"].min()

# Print the result
print(f"Epoch with the lowest G_loss: {min_g_loss_epoch}, G_loss value: {min_g_loss_value}")
# Epoch with the lowest G_loss: 38, G_loss value: 1.03392589092255
