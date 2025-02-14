# QUANTUM-ML
An implemenation of "Option Pricing Using Quantum Computers"
# Quantum Generative Adversarial Network (QGAN) for Option Pricing

This repository contains a Python implementation of a Quantum Generative Adversarial Network (QGAN) designed for pricing European call options. The code is based on the principles discussed in the research paper:

**"Option Pricing using Quantum Computers" (arXiv:1905.02666)**
*Somaya Abuhmed, Donatello Conte, Roman Gantner, Christa Zoufal, Juerg Woehr, and Stefan Woerner*

## Code Description: `main.py`

This Python script (`main.py`) demonstrates the concept of using a hybrid Quantum-Classical Generative Adversarial Network (QGAN) to learn the risk-neutral probability distribution of an underlying asset's price and subsequently use this distribution to price European call options.

### Key Components:

1.  **Parameter Setup:**
    *   Defines financial parameters for option pricing: initial stock price (`S0`), risk-neutral drift (`mu`), volatility (`sigma`), time to maturity (`T`), strike price (`strike_price`), and risk-free rate (`risk_free_rate`).
    *   Sets up parameters for the quantum circuits: number of qubits (`n_qubits`), price discretization (`price_discretization`), and number of bins (`num_bins`).
    *   Configures training parameters for the QGAN: number of real data samples (`num_real_samples`), batch size (`batch_size`), number of epochs (`num_epochs`), and shots for quantum measurements (`shots`).
    *   Initializes classical optimizers (`COBYLA`) for training the QGAN.
    *   Sets up the Qiskit Aer `AerSimulator` backend, optionally utilizing GPU acceleration if `qiskit-aer-gpu` is installed and configured correctly (environment variable `QISKIT_AER_DEVICE=GPU`).

2.  **Classical Data Generation (GBM Simulation):**
    *   Implements `simulate_asset_prices_gbm` to generate "real" asset price data at maturity using Geometric Brownian Motion (GBM). GBM is a simplified model for stock price dynamics.
    *   Implements `discretize_prices` to discretize the continuous asset prices into a set of bins, which is necessary for encoding data into quantum states.

3.  **Quantum Circuit Definitions (Generator and Discriminator):**
    *   `create_generator_circuit`: Defines a parameterized quantum circuit for the Generator. This circuit takes random noise as input and learns to generate quantum states representing samples from the risk-neutral asset price distribution. The circuit uses `RY` rotations and `CX` gates in a layered structure.
    *   `create_discriminator_circuit`: Defines a parameterized quantum circuit for the Discriminator. This circuit learns to distinguish between "real" (GBM-simulated) and "fake" (generator-created) asset price distributions. The circuit uses `RZ` and `RY` rotations and `CX` gates in a layered structure, ending with a measurement on the first qubit.

4.  **QGAN Training Functions:**
    *   `get_basis_state_vector`: Creates a basis state vector corresponding to a discretized price bin. This is used to encode "real" data into quantum states.
    *   `run_discriminator`: Executes the discriminator quantum circuit on a given input quantum state and returns the probability of measuring '1'.
    *   `qgan_loss_discriminator`: Defines the loss function for the discriminator, aiming to maximize its ability to distinguish real from fake data (Binary Cross-Entropy-like loss).
    *   `qgan_loss_generator`: Defines the loss function for the generator, aiming to minimize the discriminator's ability to distinguish fake data (i.e., to "fool" the discriminator).
    *   `train_qgan`: Implements the core QGAN training loop. It iteratively trains the discriminator and generator by optimizing their respective quantum circuit parameters using classical optimizers (`COBYLA`).

5.  **Option Pricing with Trained Generator:**
    *   `sample_asset_price_qgan`: Samples asset prices from the trained QGAN generator by extracting the probabilities from the generator's output statevector.
    *   `price_option_qgan`: Calculates the European call option price using the probability distribution learned by the QGAN. It computes the expected payoff based on this distribution and discounts it to the present value.

6.  **Classical Black-Scholes Price for Comparison:**
    *   `black_scholes_call`: Implements the Black-Scholes formula to calculate the classical option price, providing a benchmark for comparison with the QGAN's price.

7.  **Visualization:**
    *   Plots the training loss curves for both the discriminator and generator.
    *   Generates and displays histograms comparing the distribution of asset prices generated by the QGAN to the "real" data distribution from GBM simulation.

### GPU Optimization:

The code includes an option to run the quantum circuit simulations on a GPU using `qiskit-aer-gpu`. It checks for the environment variable `QISKIT_AER_DEVICE=GPU` and attempts to initialize the `AerSimulator` with GPU support if detected. This can significantly speed up simulations, especially for larger qubit counts, provided a compatible NVIDIA GPU and CUDA drivers are installed.

## Research Paper: "Option Pricing using Quantum Computers" (arXiv:1905.02666)

The research paper "Option Pricing using Quantum Computers" explores the potential of using quantum computing to improve option pricing methods.  The paper focuses on demonstrating a Quantum Generative Adversarial Network (QGAN) approach for learning the risk-neutral probability distribution of asset prices, which is a key component in derivative pricing.

### Key Ideas from the Paper:

*   **Quantum Generative Models for Finance:** The paper proposes using quantum generative models, specifically QGANs, to model financial data distributions, such as the risk-neutral distribution of asset prices.
*   **Hybrid Quantum-Classical Approach:** The QGAN framework is inherently hybrid, using quantum circuits for the generator and discriminator and classical optimization algorithms to train the parameters of these circuits.
*   **Basis Encoding for Financial Data:** The paper suggests using basis encoding to represent discretized financial data (like asset prices) as quantum states, making it compatible with quantum circuit processing.
*   **Potential for Quantum Advantage:** While the paper is primarily a proof-of-concept, it explores the potential for quantum algorithms to offer advantages in financial modeling, particularly in scenarios where classical Monte Carlo methods might be computationally expensive or less efficient.

**Limitations of the Paper and this Code Implementation:**

*   **Proof-of-Concept:** Both the paper and this code are primarily demonstrations of the *concept* of using QGANs for option pricing. They are not yet practical, production-ready option pricing tools.
*   **Simplified Models:** The code uses simplified models like Geometric Brownian Motion and basic quantum circuit architectures. Real-world financial markets and option pricing are far more complex.
*   **Classical Simulation:**  The code runs on classical simulators. True quantum advantage would require execution on actual quantum hardware, which is still under development and faces challenges like noise and limited qubit counts.
*   **Limited Scalability:** Simulating quantum circuits on classical computers is computationally expensive and does not scale efficiently to larger qubit numbers.

## Quantum Machine Learning and its Potential Impact

Quantum Machine Learning (QML) is a burgeoning field that explores how quantum computers can enhance or revolutionize machine learning tasks.  While still in its early stages, QML holds tremendous potential to impact various aspects of the world:

*   **Enhanced Computational Power:** Quantum computers, in principle, can solve certain computational problems exponentially faster than classical computers. This could lead to breakthroughs in machine learning, enabling the training of more complex models on larger datasets that are currently intractable.

*   **Improved Algorithm Efficiency:** Quantum algorithms might offer more efficient ways to perform core machine learning tasks, such as optimization, sampling, and feature extraction. This could lead to faster training times and better-performing models.

*   **New Types of Machine Learning Models:** QML could inspire the development of entirely new types of machine learning models and algorithms that leverage quantum phenomena like superposition and entanglement in ways that classical machine learning cannot.

*   **Applications Across Industries:** The potential applications of QML are vast and span across various industries:
    *   **Finance:** Faster and more accurate financial modeling, risk management, fraud detection, algorithmic trading.
    *   **Healthcare:** Drug discovery and development, personalized medicine, medical image analysis.
    *   **Materials Science:** Design of new materials with specific properties, catalyst discovery.
    *   **Artificial Intelligence:** Development of more powerful AI systems, natural language processing, computer vision.
    *   **Logistics and Optimization:** Solving complex optimization problems in supply chain management, routing, and scheduling.

**Important Note:**  Quantum Machine Learning is still a developing field.  While the theoretical potential is significant, realizing practical quantum advantages in machine learning will require further advancements in quantum hardware, algorithm development, and addressing challenges like noise and error correction in quantum computers.

This code serves as a starting point for understanding and exploring the exciting intersection of quantum computing and machine learning in the context of financial applications. As quantum computing technology progresses, QML is expected to play an increasingly important role in various domains, potentially transforming how we approach complex computational problems.

Note : you will need linux to run this since it uses the gpu and to qiskit-aer-gpu is not available on windows.
