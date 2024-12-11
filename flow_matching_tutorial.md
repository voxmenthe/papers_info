# A Tutorial on Flow Matching for Generative Modeling

Flow matching has emerged as a powerful and versatile framework for generative modeling, offering a compelling alternative to diffusion models while retaining simplicity and efficiency. This tutorial provides a comprehensive introduction to flow matching, drawing upon three recent papers: "Preference Alignment with Flow Matching," "Discrete Flow Matching," and "Optimal Flow Matching: Learning Straight Trajectories in Just One Step." We will delve into each paper, exploring their core ideas, contributions, and limitations, while also comparing and contrasting their approaches.

**I. Introduction to Flow Matching**

Flow matching aims to learn a transformation that maps a simple source distribution (e.g., Gaussian noise) to a complex target distribution (e.g., images, text). Instead of directly learning this transformation, flow matching learns a *velocity field* that guides the evolution of the source distribution towards the target distribution over a continuous time variable *t*. This velocity field is learned by regressing onto conditional velocities derived from data. The core idea is that by aligning the learned velocity field with the true underlying velocity of the data generating process, we can effectively sample from the target distribution.

**II. Continuous Flow Matching in Euclidean Space**

The foundational concept of flow matching operates in continuous Euclidean space. Consider a source distribution $p_0$ and a target distribution $p_1$. We define a probability path $p_t$, a continuous interpolation between $p_0$ and $p_1$, such that $p_{t=0} = p_0$ and $p_{t=1} = p_1$. A time-varying velocity field $u_t(x)$ governs the evolution of samples along this path. The velocity field is learned by minimizing a loss function that encourages the learned velocity to align with the direction of the data generating process, often approximated by linear interpolation between sampled data pairs $(x_0, x_1)$:

$L_{FM}(u) = \int_0^1 \int_{R^D \times R^D} ||u_t(x_t) - (x_1 - x_0)||^2 \pi(x_0, x_1) dx_0 dx_1 dt$,

where $x_t = (1-t)x_0 + tx_1$ and $\pi(x_0, x_1)$ is a joint distribution over the source and target samples.

**III. "Optimal Flow Matching: Learning Straight Trajectories in Just One Step"**

This paper addresses a key challenge in flow matching: the curvature of learned trajectories, which can lead to computational inefficiencies during sampling. It proposes Optimal Flow Matching (OFM), which restricts the learned velocity field to be the gradient of a convex function $\Psi$. This restriction guarantees straight-line trajectories and recovers the optimal transport (OT) map for the quadratic cost function in a single flow matching step. OFM minimizes the following loss:

$L_{OFM}(\Psi) = \int_0^1 \int_{R^D \times R^D} ||u^\Psi_t(x_t) - (x_1 - x_0)||^2 \pi(x_0, x_1) dx_0 dx_1 dt$,

where $u^\Psi_t(x_t)$ is the optimal velocity field derived from $\Psi$. The key innovation lies in parametrizing $\Psi$ using Input Convex Neural Networks (ICNNs), ensuring the convexity constraint. A significant advantage of OFM is its theoretical guarantee of recovering the OT map, leading to efficient sampling and improved alignment with the target distribution. However, it relies on Hessian inversion, which can be computationally expensive, and is limited by the expressiveness of ICNNs.

**IV. "Discrete Flow Matching"**

This paper extends flow matching to discrete data, crucial for applications like language modeling. It introduces a discrete flow paradigm based on continuous-time discrete Markov chains (CTMC). Similar to continuous flow matching, it learns a probability velocity that guides the transition probabilities between discrete states. The paper introduces a general framework for probability paths and derives a closed-form expression for the generating probability velocity. A key contribution is the introduction of *corrector sampling*, which combines forward and backward time sampling to refine the generated samples and improve alignment with the target distribution. This method demonstrates impressive results on code generation and language modeling tasks, significantly closing the gap between autoregressive models and discrete flow models. However, it still faces challenges in terms of sampling efficiency compared to continuous flow matching.

**V. "Preference Alignment with Flow Matching"**

This paper focuses on aligning pre-trained models with human preferences, a critical aspect of many AI applications. It proposes Preference Flow Matching (PFM), which learns a flow from less preferred data to more preferred data. PFM leverages flow-based models to transform less preferred outcomes into preferred ones, effectively aligning model outputs with human preferences without relying on explicit reward functions. This direct modeling of preference flow avoids issues like overfitting in reward models. It employs flow matching techniques and focuses on learning the flow rather than fine-tuning the pre-trained model, offering advantages in terms of scalability and accessibility, especially when dealing with black-box APIs. However,  PFM's performance in highly optimized domains (like expert demonstrations in RL) might be limited, and it assumes non-deterministic preferences for robust performance.

**VI. Comparing and Contrasting the Approaches**

| Feature | OFM | Discrete FM | PFM |
|---|---|---|---|
| Data Type | Continuous | Discrete | Continuous/Discrete (through embeddings) |
| Core Idea | Straight-line trajectories, OT map recovery | Discrete probability velocities, corrector sampling | Preference-guided flow learning |
| Key Contribution | Efficient sampling, theoretical guarantees | Non-autoregressive discrete data generation | Alignment with human preferences, black-box compatibility |
| Limitations | Hessian inversion, ICNN expressiveness | Sampling efficiency | Limited performance in optimized domains, preference assumptions |

**VII. Post-Training and Fine-Tuning**

Both "Discrete Flow Matching" and "Preference Alignment with Flow Matching" discuss post-training techniques to improve performance. "Discrete Flow Matching" introduces *corrector sampling* to refine the generated samples and improve alignment. "Preference Alignment with Flow Matching" proposes *iterative flow matching* to further enhance alignment with preferences. These post-training methods offer valuable tools for enhancing the quality of generated samples and customizing models to specific requirements.

**VIII. Applications and Future Directions**

Flow matching has seen widespread application in diverse domains, including image and video generation, molecule generation, and language modeling. Future research directions include exploring more sophisticated probability paths, improving sampling efficiency in discrete domains, and developing more expressive ICNN architectures. Furthermore, integrating flow matching with other generative modeling techniques and exploring its application in new areas holds significant promise.

This tutorial provided a high-level overview of flow matching and delved into three recent papers that advance the state-of-the-art in this field. We hope this introduction inspires you to further explore this powerful and versatile generative modeling framework.

# Deep Dive into Flow Matching Methods

This section delves deeper into the key contributions and implementation details of the three flow matching papers discussed in the previous section.

**I. "Optimal Flow Matching: Learning Straight Trajectories in Just One Step"**

**A. Background:** This paper builds upon the connection between Optimal Transport (OT) and flow matching. Recall that OT seeks the most efficient way to transport mass from one distribution ($p_0$) to another ($p_1$). For the quadratic cost, the OT map ($T^*$) transforms $p_0$ to $p_1$ along straight lines. Flow Matching, on the other hand, often results in curved trajectories. The key insight of this paper is that by restricting the velocity field to be the gradient of a convex function, we can recover straight-line trajectories and the OT map.

**B. Key Contributions:**

1. **Optimal Vector Fields:** The paper introduces the concept of *optimal vector fields*, which are defined as gradients of convex functions. This restriction ensures straight-line trajectories.
2. **Optimal Flow Matching (OFM) Loss:** The core contribution is the OFM loss, which minimizes the difference between the learned optimal vector field and the desired displacement between sampled pairs $(x_0, x_1)$:

$L_{OFM}(\Psi) = \int_0^1 \int_{R^D \times R^D} ||u^\Psi_t(x_t) - (x_1 - x_0)||^2 \pi(x_0, x_1) dx_0 dx_1 dt$

3. **ICNN Parametrization:**  The convexity constraint on the potential function $\Psi$ is enforced by parametrizing it using Input Convex Neural Networks (ICNNs).
4. **Theoretical Guarantees:** The paper provides theoretical guarantees that OFM recovers the OT map for the quadratic cost in a single flow matching step.

**C. Implementation:**

1. **ICNN Architecture:** Choose an appropriate ICNN architecture for your application. Fully-connected architectures are common, but convolutional ICNNs can also be used for image data. Ensure proper initialization for stable training (e.g., using techniques proposed in [27]).
2. **Flow Map Inversion:** A crucial step is inverting the flow map to find the initial point ($z_0$) corresponding to a given point ($x_t$). This involves solving a convex optimization problem:

$(\phi^\Psi)^{-1}(x_t) = \text{argmin}_{z_0} \frac{(1-t)}{2} ||z_0||^2 + \Psi(z_0) - (x_t, z_0)$

This can be solved efficiently using standard convex optimization techniques like LBFGS. Consider incorporating amortization strategies [2] for faster inversion.

3. **Loss Computation and Optimization:**  Calculate the OFM loss using Monte Carlo estimation and optimize using Adam or another suitable optimizer. The gradient of the loss involves Hessian inversion, which can be computationally expensive.

**II. "Discrete Flow Matching"**

**A. Background:** This paper tackles the challenge of applying flow matching to discrete data. Existing methods either embed discrete data in continuous space or rely on restrictive assumptions. This work introduces a framework that operates directly on discrete data using continuous-time discrete Markov chains.

**B. Key Contributions:**

1. **Discrete Probability Paths:** The paper introduces a general framework for probability paths in discrete space, allowing for flexible interpolation between source and target distributions.
2. **Probability Velocity:**  A core concept is the *probability velocity*, which guides the transition probabilities between discrete states. The paper derives a closed-form expression for this velocity based on learned posteriors.
3. **Corrector Sampling:** A significant contribution is *corrector sampling*, which combines forward and backward time sampling to refine generated samples and improve alignment with the target distribution.

**C. Implementation:**

1. **Posterior Estimation:** The key component is estimating the posterior probabilities $w_t^j(x^i|x_t)$ of the conditional probability path. This can be done by minimizing a cross-entropy loss.
2. **Probability Velocity Calculation:**  Once the posteriors are learned, compute the probability velocity using the closed-form expression provided in the paper. This velocity depends on the chosen probability path and scheduler.
3. **Sampling:** Generate samples by iteratively updating the discrete state according to the learned probability velocity. Consider adaptive step sizes [Campbell et al., 2024] for stable sampling.
4. **Corrector Sampling:** Implement corrector sampling by combining forward and backward time sampling steps. The choice of corrector scheduler can significantly impact the quality of generated samples.

**III. "Preference Alignment with Flow Matching"**

**A. Background:** This paper focuses on aligning pre-trained models with human preferences. Traditional methods often involve fine-tuning the model, which can be computationally expensive and inaccessible for black-box APIs. PFM proposes an alternative approach by learning a flow that maps less preferred data to more preferred data.

**B. Key Contributions:**

1. **Preference Flow:** The central idea is learning a *preference flow* that transforms less preferred outputs into preferred ones. This avoids the need for explicit reward functions, preventing overfitting in reward models.
2. **Flow Matching for Alignment:** The paper leverages flow matching techniques to learn this preference flow. A key advantage is that it aligns the model's output distribution without directly modifying the pre-trained model parameters.
3. **Iterative Flow Matching:** It introduces *iterative flow matching* to further improve the alignment by refining the flow based on the transformed data.

**C. Implementation:**

1. **Preference Data Collection:** Collect paired data of less preferred ($y^-$) and preferred ($y^+$) outputs from the pre-trained model.
2. **Flow Model Training:** Train a flow-based model to map $y^-$ to $y^+$, using a suitable flow architecture (e.g., RealNVP, Glow).
3. **Alignment:**  Given an input $x$, generate an initial output $y$ from the pre-trained model. Then, transform $y$ using the learned flow model to obtain the aligned output $y'$.
4. **Iterative Alignment:** For iterative flow matching, repeatedly apply the learned flow to the transformed output to further enhance alignment.

By following the implementation details outlined above, you can effectively apply these flow matching methods to diverse generative modeling tasks and achieve state-of-the-art results. Remember to experiment with different architectures, schedulers, and hyperparameters to optimize performance for your specific application.

# Generative Modeling Tutorial: Flow Matching and Beyond

**Part 1: Introduction and Foundations**


Welcome to the Generative Modeling Tutorial: Flow Matching and Beyond. In this workshop, we'll explore flow matching, a powerful generative modeling technique. We'll start with the foundations and then delve into advanced methods and applications.

**What is Generative Modeling?**

What is generative modeling? It's the task of learning a probability distribution from data, allowing us to generate new, similar data. Imagine having a dataset of images of cats. A generative model learns the underlying distribution of cat features, enabling us to create new, realistic cat images. Generative modeling has applications in various fields, from image synthesis and drug discovery to music composition and natural language processing. Several types of generative models exist, including Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and flow-based models, which are the focus of this workshop.

**Normalizing Flows**

Normalizing flows are a class of generative models that learn an invertible transformation from a simple base distribution (like a Gaussian) to the target data distribution. This transformation is typically parameterized by a neural network. The key idea is to use the change-of-variables formula to compute the probability density of the transformed data.

**(The Change of Variables Formula**

$p_1(y) = p_0(\phi^{-1}(y)) \left| \det \frac{\partial \phi^{-1}(y)}{\partial y} \right|$

This formula relates the density of the transformed data ($p_1$) to the density of the base distribution ($p_0$) and the Jacobian determinant of the inverse transformation.

The key idea is that when we transform data using a function, we need to account for how that transformation stretches or compresses space when computing probabilities. Let me break this down:

1. Basic Intuition:
- Consider transforming data using a function φ from space x to space y
- If φ stretches out space in some regions, the probability density needs to become smaller there (since the same probability mass is spread over a larger region)
- If φ compresses space, the density needs to become larger (same mass squeezed into smaller region)
- The Jacobian determinant precisely measures this local stretching/compression

2. The Formula:
The change of variables formula states:

$$p_1(y) = p_0(φ^{-1}(y)) \left|\det \frac{\partial φ^{-1}}{\partial y}\right|$$

Where:
- $$p_1(y)$$ is the density of the transformed data
- $$p_0(x)$$ is the density of the base distribution
- $$φ^{-1}$$ is the inverse transformation
- The determinant term corrects for the stretching/compression

3. Why the Inverse Jacobian:
- The inverse transformation $$φ^{-1}$$ tells us "where did this y point come from in x space?"
- Its Jacobian determinant measures how much the inverse transformation locally stretches/compresses space
- We need this to properly normalize the probability density in the transformed space

4. Example:
- If φ doubles distances (stretches space), its inverse $$φ^{-1}$$ halves distances
- The Jacobian determinant will be 1/2, reflecting that the density needs to be halved since the same probability mass is spread over twice the space

5. Alternative Form:
We can also write this in terms of the forward transformation:

$$p_1(y) = \frac{p_0(x)}{\left|\det \frac{\partial φ}{\partial x}\right|}$$

Both forms are equivalent, but depending on whether you're working with the forward or inverse transformation, one might be more convenient.

The Jacobian determinant is crucial because:
1. It ensures the transformed distribution remains properly normalized
2. It accounts for how the transformation locally warps space
3. It allows us to compute exact probabilities in the transformed space

This is why normalizing flows need to be invertible with tractable Jacobian determinants - we need both properties to compute exact probabilities using the change of variables formula.

**Relationship between normalizing flows and flow matching.**

Normalizing flows and flow matching are related but distinct approaches to generative modeling. The key distinction is in how they are trained:

1. Normalizing Flows:
- Train by directly maximizing the likelihood using the change of variables formula
- Requires explicitly computing Jacobian determinants during training
- Need the transformation to be invertible
- Training involves integrating ODEs when using continuous-time models

2. Flow Matching:
- Trains by directly learning the velocity field that guides points from source to target distribution
- Avoids computing Jacobian determinants during training
- No need to explicitly enforce invertibility
- Training is "simulation-free" - doesn't require integrating ODEs

The key insight of flow matching is that instead of optimizing the transformation directly (as in normalizing flows), we can learn the velocity field that describes how points should move over time to transform the source distribution into the target distribution.

Here's a concrete comparison:

Normalizing Flows:
```python
# Training involves:
log_prob = base_distribution_log_prob(inverse_transform(x)) + log_det_jacobian
loss = -log_prob  # Negative log likelihood
```

Flow Matching:
```python
# Training involves:
velocity_target = construct_target_velocity(x0, x1)
velocity_predicted = neural_network(t, xt)
loss = ||velocity_predicted - velocity_target||^2
```

The big advantage of flow matching is computational efficiency during training - it avoids expensive ODE integration and Jacobian computations. However, at inference time, both methods need to integrate the learned dynamics to generate samples.

Think of it like this:
- Normalizing flows learn the entire transformation at once by maximizing likelihood
- Flow matching learns the "directions" points should move at each instant to reach their targets

Flow matching can be seen as a clever alternative training strategy for continuous normalizing flows that makes training much more efficient, while preserving the key benefits of flow-based models.

This relationship is similar to how score matching provides an alternative way to train score-based diffusion models without having to explicitly compute normalized probabilities.


**Residual Flows**

Computing the Jacobian determinant can be computationally expensive. Residual flows simplify this by using a specific structure for the transformation:

$\phi(x) = x + g(x)$

where $g(x)$ is a neural network with a small output relative to $x$. This structure leads to an efficient computation of the Jacobian determinant.

**Continuous-Time Limit**

We can take the continuous-time limit of residual flows by considering an infinite number of infinitesimally small transformations. This leads to an ordinary differential equation (ODE) that describes the evolution of the data:

$\frac{dx_t}{dt} = u_t(x_t)$

where $u_t(x_t)$ is a time-varying velocity field. This connection to ODEs is crucial for flow matching.

**Flow Matching**

Flow matching offers a simulation-free way to train these continuous normalizing flows. Instead of maximizing likelihood directly, which requires integrating the ODE, we learn the velocity field $u_\theta$ parameterized by $\theta$ by regressing onto a target velocity.

**(Equation on Slide: Flow Matching Loss**

$L_{FM}(u_\theta) = \mathbb{E}_{t, x_0 \sim p_0, x_1 \sim p_1} [||u_\theta(t, x_t) - u_t^*(x_t)||^2]$

Here, $u_t^*(x_t)$ represents a target velocity often chosen as the linear interpolation velocity $(x_1 - x_0)$. The flow matching models learn by minimizing the difference between the parametrized velocity $u_\theta$ and the target $u_t^*$. This offers several advantages like avoiding the expensive ODE simulations during training, a more direct optimization of the velocity field, and flexible choices for probability paths, which we'll discuss later.

**Part 2: Deep Dive into Flow Matching Methods**

**Conditional Flow Matching (CFM)**

Conditional flow matching introduces the idea of learning the velocity field conditioned on a target sample $x_1$. This is particularly useful when the target distribution is complex.

**Equation: Marginal Probability Path**

$p_t(x) = \int p_t(x | x_1) p_1(x_1) dx_1$

This equation defines the probability path as a marginalization over the conditional probability path.

**Gaussian Probability Paths**

A common choice for the conditional path is a Gaussian distribution. This simplifies the calculations and allows for closed-form solutions in some cases.

**Equation: Conditional Velocity Field**

$u_t(x | x_1) = \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \dot{\mu}_t(x_1)$

This velocity field induces a Gaussian probability path, where the time derivatives of the mean ($\dot \mu$) and standard deviation ($\dot \sigma$) guide the trajectories.



**CFM Loss and its Advantages**

The CFM loss function allows us to train the model without needing to explicitly compute the marginal velocity field:

$L_{CFM}(\theta) = \mathbb{E}_{t, x_0 \sim p_0, x_1 \sim p_1} [||u_\theta(t, x_t | x_1) - u_t^*(x_t| x_1)||^2]$

By marginalizing over $x_1$, the learned conditional velocity field can generate the entire marginal probability path $p_t(x)$. This also offers a more stable training process due to the lower variance of the conditional velocity estimates compared to their marginal counterparts. The concept of *couplings*, including the optimal transport coupling which minimizes the distance between distributions in transport, can be used to construct training pairs in CFM. This can avoid intersecting trajectories during training.

**Optimal Flow Matching (OFM)**

A challenge in flow matching is the potential curvature of the learned trajectories. OFM tackles this by restricting the velocity field to be the gradient of a convex potential function:

$u_t^\Psi(x) = \nabla \Psi((\phi_t^\Psi)^{-1}(x)) - (\phi_t^\Psi)^{-1}(x)$

This ensures straight-line trajectories and leads to efficient sampling.

**OFM Loss and ICNNs**

OFM minimizes a similar loss to CFM, but uses optimal vector fields:

$L_{OFM}(\Psi) = \int_0^1 \int_{R^D \times R^D} ||u^\Psi_t(x_t) - (x_1 - x_0)||^2 \pi(x_0, x_1) dx_0 dx_1 dt$.

The convexity of $\Psi$ is ensured by parameterizing it with Input Convex Neural Networks (ICNNs).

**Flow Map Inversion and Hessian**

Implementing OFM involves two key challenges: inverting the flow map $(\phi_t^\Psi)^{-1}(x)$ and inverting the Hessian $\nabla^2 \Psi$. Both operations can be computationally expensive, especially in high dimensions. Amortization strategies can be applied for efficient flow map inversion. Hessian inversion remains a challenge and an active area of research.

**Discrete Flow Matching**

Extending flow matching to discrete data is crucial for applications like language modeling. Discrete Flow Matching uses Continuous Time Markov Chains (CTMCs) to define probability flows in discrete spaces.

**Probability Velocity**

The key concept in Discrete Flow Matching is *probability velocity*. This velocity governs the evolution of the probability distribution over the discrete states. A closed-form expression for the generating probability velocity is provided in "Discrete Flow Matching," allowing for efficient computation.

**Corrector Sampling**

*Corrector sampling* is an important technique introduced in "Discrete Flow Matching". It combines forward and backward sampling steps to improve the quality of generated samples and better align the learned distribution with the true data distribution. The choice of appropriate forward, backward, and corrector schedulers greatly affects the performance.

**Preference Alignment with Flow Matching (PFM)**

PFM focuses on aligning a pre-trained model with human preferences without altering the pre-trained parameters. It learns a transformation from less preferred to more preferred samples.

**PFM: Learning the Flow**

Instead of directly fine-tuning the pre-trained model, PFM learns a separate flow model that transforms the output of the pre-trained model. This allows for efficient alignment without affecting the original model's capabilities.

**Iterative Flow Matching in PFM**

Iterative Flow Matching further refines the alignment in PFM by repeatedly applying the learned flow to the transformed samples. This allows for continuous improvement and better alignment with preferences. PFM is particularly useful when fine-tuning is not possible, for example with black-box APIs. However, it relies on non-deterministic preferences and requires an adequate method for embedding data if it works with non-continuous data like text.

**Part 3: Advanced Techniques and Applications**

**Advanced Sampling Techniques**

We've already encountered corrector and iterative sampling. Let's delve deeper. These techniques refine the sampling process and are especially relevant in complex scenarios. Corrector sampling helps with the discrete case, while iterative flow matching is crucial for preference alignment.

**Applications: Image & Video, Molecules, Language**

Flow matching has shown promise in diverse applications. In images and videos, it allows for high-fidelity generation. In molecule design, equivariant flow matching incorporates molecular symmetries. In language modeling, discrete flow matching handles the discrete nature of text, and PFM aligns the generated text with human preferences.

**Iterative Corrupted Trajectory Matching**

ICTM offers an efficient way to use flow models as priors for solving linear inverse problems, which arise in various domains like medical imaging and astronomy. Traditional approaches to incorporating flow priors into MAP estimation can be computationally expensive due to ODE solver backpropagation.

**ICTM Algorithm**

ICTM approximates the MAP solution by iteratively optimizing a series of “local” MAP objectives along the flow trajectory. This avoids the expensive ODE backpropagation, making it computationally more efficient. The gradient of the local prior is efficiently calculated using Tweedie's formula.

**ICTM Applications**

ICTM demonstrates good performance in various inverse problems such as denoising, inpainting, super-resolution, deblurring, and compressed sensing. This makes it a powerful tool for high-quality image reconstruction and other downstream tasks. It’s important to note that ICTM assumes optimal transport interpolation for theoretical guarantees and is currently applicable to linear inverse problems.

This detailed walkthrough of the first three parts provides a solid foundation for understanding and implementing flow matching. Part 4 will then focus on the practical coding examples and open discussion. Remember to engage the audience with questions and encourage interaction throughout the tutorial.
