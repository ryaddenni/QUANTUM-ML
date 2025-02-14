
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from scipy.stats import norm
from qiskit.circuit import ParameterVector, Parameter
import time
import os

# --- 1. Parameter Setup ---
n_qubits = 3
num_bins = 2**n_qubits
price_range = [50, 150]
price_discretization = np.linspace(price_range[0], price_range[1], num_bins)

S0 = 100
mu = 0.05
sigma = 0.2
T = 1
strike_price = 100
risk_free_rate = 0.02

num_real_samples = 1024
batch_size = 64
num_epochs = 100
shots = 1024

optimizer_generator = COBYLA(maxiter=100)
optimizer_discriminator = COBYLA(maxiter=100)

# --- 1b. GPU Setup ---
USING_GPU = os.environ.get('QISKIT_AER_DEVICE') == 'GPU'

if USING_GPU:
    try:
        backend = AerSimulator(method='statevector', device='GPU')
        print("GPU detected! Using GPU for simulations.")
    except Exception as e:
        print(f"GPU setup failed: {e}. Falling back to CPU. Error: {e}")
        backend = AerSimulator(method='statevector')
        USING_GPU = False
else:
    print("QISKIT_AER_DEVICE not set to 'GPU'.  Using CPU simulations.")
    backend = AerSimulator(method='statevector')

# --- 3b. Setup discriminator and transpile outside the training loop ---
num_generator_params = n_qubits * 2
num_discriminator_params = 2 * n_qubits * 2

# --- 3. Quantum Circuit Definitions ---

def create_generator_circuit(n_qubits, num_params):
    qc = QuantumCircuit(n_qubits, name='generator')
    params = ParameterVector('gen_params', num_params)

    param_index = 0
    for layer in range(2):
        for qubit in range(n_qubits):
            qc.ry(params[param_index % num_params], qubit)
            param_index += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)
    return qc

def create_discriminator_circuit(n_qubits, num_params):
    qc = QuantumCircuit(n_qubits, 1, name='discriminator')
    params = ParameterVector('disc_params', num_params)

    param_index = 0
    for layer in range(2):
        for qubit in range(n_qubits):
            qc.rz(params[param_index % num_params], qubit)
            qc.ry(params[(param_index + 1) % num_params], qubit)
            param_index += 2
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)

    qc.measure(0, 0)
    return qc
generator = create_generator_circuit(n_qubits, num_generator_params)
discriminator = create_discriminator_circuit(n_qubits, num_discriminator_params)

# Transpile discriminator circuit once
transpiled_discriminator = transpile(discriminator, backend, optimization_level=3)

initial_generator_params = np.random.rand(num_generator_params)
initial_discriminator_params = np.random.rand(num_discriminator_params)

# --- 2. Helper Functions ---
def simulate_asset_prices_gbm(S0, mu, sigma, T, num_samples):  # Added S0, mu, sigma, T
    dt = T
    prices = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, num_samples))
    return prices

def discretize_prices(prices, price_discretization):
    bin_indices = np.digitize(prices, price_discretization) - 1
    bin_indices = np.clip(bin_indices, 0, len(price_discretization) - 1)
    return bin_indices

# Pre-calculate basis state vectors for efficiency
def get_basis_state_vector(bin_index, n_qubits):
    state_vector = np.zeros(2**n_qubits)
    state_vector[bin_index] = 1.0
    return state_vector

# --- 4. Run Discriminator (Optimized) ---
def run_discriminator(transpiled_discriminator, input_state_vector, discriminator_params, backend, shots):
    # Assign parameters to transpiled discriminator
    param_dict = dict(zip(transpiled_discriminator.parameters, discriminator_params))
    bound_discriminator = transpiled_discriminator.assign_parameters(param_dict)

    num_qubits = transpiled_discriminator.num_qubits #Correct Code
    input_qc = QuantumCircuit(num_qubits) # Initialize with input state
    input_qc.initialize(input_state_vector, range(num_qubits))

    qc = input_qc.compose(bound_discriminator, front=True)

    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    prob_1 = counts.get('1', 0) / shots

    return prob_1

# --- 5. Loss Functions ---
def qgan_loss_discriminator(output_real, output_fake):
    output_real = np.array(output_real) #Convert to Numpy Array
    output_fake = np.array(output_fake)  # Convert to NumPy array
    loss_real = -np.log(np.clip(output_real, 1e-15, 1))
    loss_fake = -np.log(np.clip(1 - output_fake, 1e-15, 1))
    return np.mean(loss_real + loss_fake)

def qgan_loss_generator(output_fake):
    output_fake = np.array(output_fake)
    return -np.mean(np.log(np.clip(output_fake, 1e-15, 1)))

# --- 6. Training Function ---
def train_qgan(generator, discriminator, initial_generator_params, initial_discriminator_params,
               real_data_bins, num_epochs, batch_size, optimizer_generator, optimizer_discriminator, backend,
               qgan_loss_discriminator, qgan_loss_generator, transpiled_discriminator):

    generator_params = initial_generator_params.copy()
    discriminator_params = initial_discriminator_params.copy()

    history = {'generator_loss': [], 'discriminator_loss': []}

    #Precompute real state vector for the batch
    real_state_vectors = [get_basis_state_vector(i,n_qubits) for i in real_data_bins]

    for epoch in range(num_epochs):
        start_time = time.time() #Record start time
        np.random.shuffle(real_data_bins)

        # Pre-generate fake data points once per epoch
        fake_data_points = np.random.rand(batch_size, num_generator_params)

        for batch_start in range(0, len(real_data_bins), batch_size):
            batch_real_bins = real_data_bins[batch_start:batch_start + batch_size]

            #Batch training for Real for batch_start
            num_items = len(batch_real_bins) #Correct Code
            disc_out_reals = [] #Output

            #Iterate to get all basis state vectors in all batch real bins
            for i in range(num_items): #Correct Batch
                bin_index = batch_real_bins[i] #Correct batch
                real_state_vector = real_state_vectors[bin_index] #Correct Code
                prob_real = run_discriminator(transpiled_discriminator, real_state_vector, discriminator_params, backend, shots) #Re-Use variable

                #For logging purposes
                disc_out_reals.append(prob_real)

            def discriminator_loss_function(current_discriminator_params):
                discriminator_output_fake_batch = []

                # --- 2. Train Generator ---
                num_items = len(fake_data_points) #Correct Code
                for i in range(num_items): #Correct
                    noise_vector = fake_data_points[i] #Correct index

                    bound_generator = generator.assign_parameters(dict(zip(generator.parameters, noise_vector))) #Correct code
                    generator_state = Statevector(bound_generator) #Create Statevector

                    prob_fake = run_discriminator(transpiled_discriminator, generator_state, current_discriminator_params, backend, shots) #Change code
                    discriminator_output_fake_batch.append(prob_fake) #Append batch

                loss_val = qgan_loss_discriminator(disc_out_reals, discriminator_output_fake_batch)
                return loss_val

            # Optimize discriminator parameters to minimize the discriminator loss
            discriminator_params = optimizer_discriminator.minimize(discriminator_loss_function, discriminator_params).x

            # --- 3. Train Generator ---
            def generator_loss_function(current_generator_params):
                discriminator_output_fake_batch = []
                # batch fake samples
                for i in range(num_items): #Correct Loop
                    noise_vector = fake_data_points[i] #Get fake sample

                    bound_generator = generator.assign_parameters(dict(zip(generator.parameters, noise_vector))) #Apply random values to the state
                    generator_state = Statevector(bound_generator) #Create Statevector

                    #Use same variables
                    prob_fake = run_discriminator(transpiled_discriminator, generator_state, discriminator_params, backend, shots) #Run discriminate, Change code
                    discriminator_output_fake_batch.append(prob_fake) #Use batch
                loss_val = qgan_loss_generator(discriminator_output_fake_batch) #Loss code

                return loss_val

            #Optimize Generator parameters to minimize discriminator loss (fooling discriminator)
            generator_params = optimizer_generator.minimize(generator_loss_function, generator_params).x #Change code

            # --- Logging ---
            discriminator_loss_val = discriminator_loss_function(discriminator_params)
            generator_loss_val = generator_loss_function(generator_params)

            history['discriminator_loss'].append(discriminator_loss_val)
            history['generator_loss'].append(generator_loss_val)

            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_start//batch_size + 1}/{len(real_data_bins)//batch_size + 1}, "
                  f"D Loss: {discriminator_loss_val:.4f}, G Loss: {generator_loss_val:.4f}")

        end_time = time.time() #Record end time
        epoch_time = end_time - start_time #Calculate epoch
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} seconds")

    return generator, discriminator, generator_params, discriminator_params, history

# --- 8. Option Pricing and Evaluation ---
def price_option_qgan(trained_generator, generator_params, num_generator_params, strike_price, risk_free_rate, time_to_maturity, num_samples_option_pricing, backend, price_discretization):
    """Calculate the option price given the generated probabilities"""
    
    # Get the probabilities distribution (already normalized)
    probabilities = sample_asset_price_qgan(trained_generator, generator_params, backend, price_discretization)
    
    #Handle for the error
    if isinstance(probabilities, int) and probabilities == -1: #New error with type check
        return probabilities

    #Scale it, and normalize
    num_bins = len(price_discretization)
    
    #Calculations, make sure that is not happening in loops
    payoffs = np.maximum(price_discretization - strike_price, 0)
    expected_payoff = np.sum(payoffs * probabilities) #dot product
    option_price_qgan = np.exp(-risk_free_rate * time_to_maturity) * expected_payoff

    return option_price_qgan

def sample_asset_price_qgan(trained_generator, generator_params, backend, price_discretization):
    """Sample asset prices by extracting state probabilities directly."""
    try:
        num_qubits = trained_generator.num_qubits
    
        # Assign parameters to the trained generator
        param_dict = dict(zip(trained_generator.parameters, generator_params))
        bound_generator = trained_generator.assign_parameters(param_dict)
        
        # Get the statevector
        statevector = Statevector(bound_generator).data

        if isinstance(statevector, np.ndarray):
            # Calculate probabilities from the statevector
            probabilities = np.abs(statevector)**2  # Probabilities are magnitude squared
            return probabilities
        else:
            print("Statevector generation failed! not numpy array object")
            return -1 #Not run
    except Exception as e:
         print(f"Sampling the asset price exception {e}!")
         return -1 #Didnt Run

def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


num_generator_params = n_qubits * 2
num_discriminator_params = 2 * n_qubits * 2

generator = create_generator_circuit(n_qubits, num_generator_params)
discriminator = create_discriminator_circuit(n_qubits, num_discriminator_params)

# Transpile discriminator circuit once
transpiled_discriminator = transpile(discriminator, backend, optimization_level=3)

initial_generator_params = np.random.rand(num_generator_params)
initial_discriminator_params = np.random.rand(num_discriminator_params)

# --- 7. Training Execution (Run to Train!) ---
print("Generating Real Data...")
real_asset_prices = simulate_asset_prices_gbm(S0, mu, sigma, T, num_real_samples)
real_data_bins = discretize_prices(real_asset_prices, price_discretization)

print("Starting QGAN Training...")
start_time = time.time()  # Record start time

trained_generator, trained_discriminator, trained_generator_params, trained_discriminator_params, training_history = train_qgan(
    generator, discriminator, initial_generator_params, initial_discriminator_params,
    real_data_bins, num_epochs, batch_size, optimizer_generator, optimizer_discriminator, backend,
    qgan_loss_discriminator, qgan_loss_generator, transpiled_discriminator
)

end_time = time.time()  # Record end time
print(f"\n--- QGAN Training Finished in {end_time - start_time:.2f} seconds ---")

# --- 8. Save the trained parameters for later use (optional) ---
#np.save('trained_generator_params.npy', trained_generator_params)
#np.save('trained_discriminator_params.npy', trained_discriminator_params)

# --- 9. Option Pricing and Evaluation ---
num_option_pricing_samples = 1000
option_price_qgan = price_option_qgan(trained_generator, trained_generator_params, num_generator_params, strike_price, risk_free_rate, T, num_option_pricing_samples, backend, price_discretization)
print(f"\nQGAN Option Price: {option_price_qgan:.4f}")

# --- 10. Classical Black-Scholes Price for Comparison ---
bs_price = black_scholes_call(S0, strike_price, T, risk_free_rate, sigma)
print(f"Black-Scholes Price: {bs_price:.4f}")

# --- 11. Visualization ---
plt.figure(figsize=(12, 5))

# Loss Curves Plot
plt.subplot(1, 2, 1)
plt.plot(training_history['discriminator_loss'], label='Discriminator Loss')
plt.plot(training_history['generator_loss'], label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('QGAN Training Loss')
plt.legend()
plt.grid(True)

# Generated Price Distribution Histogram Plot
plt.subplot(1, 2, 2)
generated_prices = []
num_histogram_samples = 1000
probabilities = sample_asset_price_qgan(trained_generator, trained_generator_params, backend, price_discretization)

if isinstance(probabilities, int) and  probabilities == -1: #Fix the error
    generated_prices = [0] * num_histogram_samples
else:
    sampled_bin_index = np.random.choice(len(price_discretization), size=num_histogram_samples, replace=True, p=probabilities)
    generated_prices.extend(price_discretization[sampled_bin_index])

plt.hist(generated_prices, bins=price_discretization, alpha=0.7, label='QGAN Generated Prices')
plt.hist(real_asset_prices, bins=price_discretization, alpha=0.5, label='Real Prices (GBM)', density=True)
plt.xlabel('Asset Price at Maturity')
plt.ylabel('Density (or Frequency)')
plt.title('Distribution of Generated Asset Prices vs. Real Data (QGAN)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
