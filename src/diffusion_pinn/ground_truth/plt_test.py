import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

# Create some simple data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create and save a simple plot
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Test Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')

# Save the plot
plt.savefig('test_plot.png')
plt.close()

print("Plot has been saved to test_plot.png")
