import matplotlib.pyplot as plt

# Example data
x = [3, 5, 7, 9, 11]  # X-axis values
y = [98.87, 97.57,96.84 , 96.58, 95.83]  # Y-axis values

# Create a line graph
plt.plot(x, y, label='Accuracy Line', marker='o')

# Add labels and title
plt.xlabel('K Values')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy Based on K Values')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Display the graph
plt.show()
