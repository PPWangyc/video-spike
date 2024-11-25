import cebra
import numpy as np
timesteps = 5000
neurons = 50
out_dim = 8

neural_data = np.random.normal(0,1,(timesteps, neurons))
print(neural_data.shape)
single_cebra_model = cebra.CEBRA(batch_size=512,
                                 output_dimension=out_dim,
                                 max_iterations=10000,
                                 max_adapt_iterations=10)
print(single_cebra_model)
single_cebra_model.fit(neural_data)
single_cebra_model.save('cebra.pt')
embedding = single_cebra_model.transform(neural_data)
assert(embedding.shape == (timesteps, out_dim))
ax = cebra.plot_embedding(embedding,None)
fig = ax.figure
fig.savefig('embedding.png')
exit()
# Use Cebra's plot_loss function to get the axes object for the loss curve
ax = cebra.plot_loss(single_cebra_model)

# Get the figure from the axes object
fig = ax.figure

# Save the figure using the savefig method from the figure object
fig.savefig('loss_curve.png')  # Save the plot to a file
