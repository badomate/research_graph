

# --- Learning Sparse Graphon Mean Field Games ---

**Christian Fabian**

Technische Universität Darmstadt

**Kai Cui**

Technische Universität Darmstadt

**Heinz Koeppl**

Technische Universität Darmstadt

## Abstract

Although the field of multi-agent reinforcement learning (MARL) has made considerable progress in the last years, solving systems with a large number of agents remains a hard challenge. Graphon mean field games (GMFGs) enable the scalable analysis of MARL problems that are otherwise intractable. By the mathematical structure of graphons, this approach is limited to dense graphs which are insufficient to describe many real-world networks such as power law graphs. Our paper introduces a novel formulation of GMFGs, called LPGMFGs, which leverages the graph theoretical concept of  $L^p$  graphons and provides a machine learning tool to efficiently and accurately approximate solutions for sparse network problems. This especially includes power law networks which are empirically observed in various application areas and cannot be captured by standard graphons. We derive theoretical existence and convergence guarantees and give empirical examples that demonstrate the accuracy of our learning approach for systems with many agents. Furthermore, we extend the Online Mirror Descent (OMD) learning algorithm to our setup to accelerate learning speed, empirically show its capabilities, and conduct a theoretical analysis using the novel concept of smoothed step graphons. In general, we provide a scalable, mathematically well-founded machine learning approach to a large class of otherwise intractable problems of great relevance in numerous fields.

Examples include neurons in the human brain (Avena-Koenigsberger et al. (2018), Bullmore and Sporns (2009), Bullmore and Sporns (2012), Sporns (2022)), people trading on a stock market (Bakker et al. (2010), Bian et al. (2016)) or the spreading of contagious diseases among citizens of a society (Newman (2002), Pastor-Satorras et al. (2015)). Due to their complexity, these systems are in general hard to model and are often controlled by using multi-agent reinforcement learning (MARL). In the last years, the field of MARL has experienced significant progress, see Canese et al. (2021) or Yang and Wang (2020) for an overview, but crucial open problems remain. While many approaches provide sound empirical results, they often lack a solid theoretical foundation (Zhang et al. (2021)). Furthermore, as the number of agents in the system increases, numerous MARL algorithms become computationally expensive and are thereby hardly scalable.

Mean field games (MFGs) (Carmona and Delarue (2018a), Carmona and Delarue (2018b)) have proven to be a valid approach for achieving both scalability and theoretical guarantees in multi-agent systems. Since they were introduced independently by Huang et al. (2006) and Lasry and Lions (2007) to address game theoretic challenges, they have become a major interest in various research fields. Extensions of the original MFG model include discrete-time formulations (Cui and Koeppl (2022), Saldi et al. (2018)), variants with major and minor agents (Carmona and Zhu (2016), Firoozi et al. (2020), Nourian and Caines (2013)) as well as zero-sum games (Choutri and Djehiche (2019), Choutri et al. (2019)). MFGs are based on the weak interaction principle where each individual has a negligible influence on the whole system. Besides the numerical and theoretical benefits of this principle, MFGs provide the modelling framework for various applications, such as autonomous driving (Huang et al. (2020)), cyber security (Kolokoltsov and Malafeyev (2018)), big data architectures (Castiglione et al. (2014)), and systemic risk in financial markets (Carmona et al. (2015), Elie et al. (2020a)). There is also some work that aims to apply MFGs to real world tasks, e.g. social networks (Yang et al. (2018)) or swarm robotics (Elamvazhuthi and Berman (2019), Cui et al. (2022)), but this field largely remains to be developed. Although our paper is of theoretical nature, its goal is to make MFGs more realistic, as we discuss below.

## 1 INTRODUCTION

In various research areas, scientists are confronted with systems of many interacting individuals or components.

Both from the classical equilibrium learning perspective and the reinforcement learning (RL) perspective, MFGs are able to provide solutions for numerous challenges where other equilibrium learning or MARL algorithms become computationally intractable. Here, learning refers to both classical computation of equilibria and RL – also known as approximate optimal control (Bertsekas (2019)) – with focus on solving complex control problems without knowing or using the model. Current RL research is addressing the approximation of Nash equilibria for stationary games (Subramanian and Mahajan (2019)), under the occurrence of noise (Carmona et al. (2019)), using entropy regularization (Anahtarci et al. (2020), Cui and Koepl (2021), Guo et al. (2022)), leveraging Fictitious Play (Cardaliaguet and Hadikhanloo (2017), Delarue and Vasileiadis (2021), Hadikhanloo and Silva (2019), Mguni et al. (2018), Perrin et al. (2021a), Perrin et al. (2021b), Perrin et al. (2020)), and increasing the robustness and efficiency of learning algorithms in general (Guo et al. (2019), Guo et al. (2020)). A learning scheme of particular interest for our paper is Online Mirror Descent (OMD) (Orabona et al. (2015), Srebro et al. (2011)). RL research has leveraged OMD to learn MFGs (Hadikhanloo (2017), Laurière et al. (2022), Perolat et al. (2021)) which ensures algorithmic scalability.

For learning applications, decision-making on graphs appears to be particularly interesting. Here, we refer to agents connected via graphical edges, as opposed to agents with states on a graph as considered e.g. in Li et al. (2019). Apart from direct graphical decompositions in MARL Qu et al. (2020), there has been recent research interest in MFGs on graphs. In classical MFGs each agent weakly interacts with all other agents at once which seems to be an unrealistic modelling assumption for many applications. To overcome this concern, graphon mean field games (Aurell et al. (2021), Caines and Huang (2019), Carmona et al. (2022), Gao and Caines (2017), Gao et al. (2021)) (GMFGs) provide a well-established tool to model games with a graphical structure. For example, Tangpi and Zhou (2022) apply GMFGs to model investment decisions in a financial market. Also based on GMFGs, Aurell et al. (2022) develop models on epidemics and provide the corresponding machine learning methods for outcome estimation. So far, most of the literature has focused on MFGs on dense graphs. To the best of our knowledge, there are only a few papers that consider sparse graphs. While Gkogkas and Kuehn (2022) focuses on Kuramoto-like models, Lacker and Soret (2022) is concerned with linear-quadratic stochastic differential games. Finally, Bayraktar et al. (2020) considers systems on not-so-dense graphs but without control and without leveraging  $L^p$  graphons. By utilizing  $L^p$  graphons (Borgs et al. (2018), Borgs et al. (2019)), our paper’s aim is to provide a general framework for learning MFGs on sparse graphs.

Sparse power law graphs (Barabási and Albert (1999), Barabási et al. (1999)) are of great interest for various

research applications such as social networks (Aparicio et al. (2015)), software engineering (Concas et al. (2007), Louridas et al. (2008), Wheeldon and Counsell (2003)), finance (D’Arcangelis and Rotundo (2016)), or biology (Nosonovsky and Roy (2020)). For more examples, see Newman (2018). Although there is strong empirical evidence for power law graphs in many research fields as mentioned above, GMFGs cannot capture these naturally sparse structures. LPGMFGs and the corresponding learning methods presented in our paper provide a novel ML tool to solve such real-world problems that are otherwise intractable. Our contributions can be summarized as follows: (i) introducing MFGs on  $L^p$  Graphons (LPGMFGs) which formalize MFGs on sparse graphs; (ii) conducting a theoretical analysis of LPGMFGs that includes the existence of equilibria as well as convergence guarantees; (iii) evaluating LPGMFGs on different examples empirically, especially in a multi-class agent setup; (iv) adapting the OMD learning scheme to our setup and thereby accelerating learning speed; (v) conducting both an empirical and a theoretical convergence analysis for the adapted OMD algorithm. Thus, our paper provides a scalable, mathematically well founded approach for learning MARL problems on sparse graphs on the theoretical side. On the practical side, different empirical examples demonstrate the scalability of the learning method and give an impression of how models on sparse graphs are often more realistic than dense networks.

## 2 $L^p$ GRAPHONS

**Central Concepts.** In this section, we briefly introduce the concept of  $L^p$  graphons pioneered by Borgs et al. (2018) and Borgs et al. (2019) which provide more details.  $L^p$  graphons can be informally thought of as adjacency matrices for graphs with (almost) infinitely many nodes. Naturally, approximating sparse finite graphs by these  $L^p$  graphons leads to the loss of some topological information, see e.g. Borgs et al. (2018). Nevertheless,  $L^p$  graphons provide an expressive asymptotic approximation of the finite case which we show both theoretically and empirically on the next pages. In contrast to standard graphons which are limited to dense graphs,  $L^p$  graphons are more general and also apply to sparse graphs. An  $L^p$  graphon is a symmetric, integrable function  $W: [0, 1]^2 \to \mathbb{R}$  with  $\|W\|_p < \infty$  where the  $L^p$  norm on graphons is  $\|W\|_p := \left( \int_{[0,1]^2} |W(x, y)|^p \, dx \, dy \right)^{1/p}$  for  $1 \le p < \infty$  and the essential supremum if  $p = \infty$ .

To quantify whether a graphon is a good approximation for a sequence of finite graphs, we associate every finite graph  $G = (V(G), E(G))$  with a graphon  $W^G$ . For a graph  $G$  with  $N$  nodes, we partition the unit interval  $[0, 1]$  into  $N$  intervals  $I_1, \dots, I_N$  of equal length. Then, the function  $W^G$  is assigned a constant value on each square  $I_i \times I_j$  ( $i, j \in V(G)$ ) which is equal to one if there is an edge between

the nodes  $i, j$  in  $G$  and zero otherwise. Thus,  $W^G$  is by construction a step-function and therefore often called step-graphon. To compare some graph  $G$  and graphon  $W$ , we can simply compare the graphons  $W$  and  $W^G$  in the space of graphons. Instead of  $W^G$  itself, we frequently consider the normalized associated graphon  $W_G / \|G\|_1$  to derive results that also hold in the sparse case. To measure how close two graphons are, the cut norm is a natural candidate and possesses many useful properties, see e.g. Lovász (2012) for a detailed discussion. For a graphon  $W: [0, 1]^2 \to \mathbb{R}$ , define the cut norm by

$$\|W\|_{\square} := \sup_{S, T \subseteq [0, 1]} \left| \int_{S \times T} W(x, y) \, dx \, dy \right|,$$

where  $S$  and  $T$  range over the measurable subsets of  $[0, 1]$ .

Starting with a graphon, we can use the following well-established construction to generate sparse random graphs with  $N$  nodes where we assume without loss of generality that the vertices are labeled by  $1, \dots, N$ . We choose  $x_1, \dots, x_N$  i.i.d. uniformly at random in  $[0, 1]$  and fix some  $\rho > 0$ . For all vertex pairs  $1 \le i < j \le N$  there is an edge between  $i$  and  $j$  with probability  $\min\{\rho W(x_i, x_j), 1\}$  which yields a sparse random graph  $\mathbf{G}(N, W, \rho)$ . A sequence of sparse random graphs generated by this method converges to the generating graphon  $W$  in the cut norm, see (Borgs et al., 2019, Theorem 2.14).

**Assumption 1.** *The sequence of normalized step-graphons  $(W_N)_{N \in \mathbb{N}}$  converges in cut norm  $\|\cdot\|_{\square}$  or equivalently in operator norm  $\|\cdot\|_{L_{\infty} \to L_1}$  (see Lovász (2012)) as  $N \to \infty$  to some graphon  $W \in \mathcal{W}_0$ , i.e.*

$$\|\rho_N^{-1} W_N - W\|_{\square} \to 0, \quad \|\rho_N^{-1} W_N - W\|_{L_{\infty} \to L_1} \to 0. \quad (1)$$

The limiting graphon in Assumption 1 is only guaranteed to exist for so-called  $L^p$  upper regular graph sequences, for details see Borgs et al. (2018), Borgs et al. (2019). This implicit assumption means that our approach does not apply to arbitrary sequences of sparse random graphs. If the average degree in the graph sequence does not tend to infinity as  $N \to \infty$ ,  $L^p$  upper regularity is not fulfilled. Thus, for example, the asymptotic behavior of ultra-sparse graph sequences is beyond the scope of  $L^p$  graphons. Nevertheless,  $L^p$  graphons are the limit of crucial sparse graph sequences such as power law graphs which cannot be provided by standard graphons. In our paper, 'power law' refers to the tail of the distribution. Figure 1 shows the advantages of  $L^p$  graphons over standard graphons using an exemplary real-world network (data from Rozemberczki et al. (2019)). The examples in the next sections are usually based on power law graphons  $W: [0, 1]^2 \to \mathbb{R}$  with  $W(x, y) = (1 - \alpha)^2(xy)^{-\alpha}$  where  $\alpha \in (0, 1)$ , see Borgs et al. (2018) for details.

**Smoothing Step Graphons.** For the theoretical analysis of the OMD algorithm in the next sections, we introduce the concept of smoothed step graphons which is new to the best of our knowledge. The basic idea is to smooth the borders of the steps. Then, the smoothed step graphon is Lipschitz continuous but still close to the original step graphon as we decrease the width  $\xi$  of the border regions.

Consider an arbitrary step graphon  $W_s$  on the unit interval partitioned into  $M$  subintervals  $\mathcal{I}_1, \dots, \mathcal{I}_M$  of equal length  $1/M$  such that  $W_s(x, y) = w_{i,j} \ge 0$  for all  $(x, y) \in \mathcal{I}_i \times \mathcal{I}_j$ ,  $1 \le i, j \le M$ . Then, for an arbitrary but fixed  $0 < \xi < 1/(2M)$ , we define the corresponding smoothed step graphon  $W_{s,\xi}$  as follows. For all  $(x, y) \in \{(x, y) \in [0, 1]^2 : (x \le \xi) \lor (x \ge 1 - \xi)\}$  and  $(x, y) \in \{(y \le \xi) \lor (y \ge 1 - \xi)\}$  we define  $W_{s,\xi}(x, y) := W_s(x, y)$ . The values of the two graphons are also defined to be identical for  $(x, y)$  with  $(x, y) \in \tilde{\mathcal{I}}_i \times \tilde{\mathcal{I}}_j$  for some  $1 \le i, j \le M$  where  $\tilde{\mathcal{I}}_i := [(i-1)/M + \xi, i/M - \xi]$  for all  $1 \le i \le M$ . In contrast to that, if  $x \in \tilde{\mathcal{I}}_i$  and  $y \in [j/M - \xi, j/M + \xi]$  for some  $1 \le i, j \le M$ , we have  $W_{s,\xi}(x, y) := \left(\frac{1}{2} - \frac{y-j/M}{2\xi}\right)w_{i,j} + \frac{y-j/M+\xi}{2\xi}w_{i,j+1}$  and analogously for  $x$  and  $y$  with switched roles. Finally, if both  $x \in [i/M - \xi, i/M + \xi]$  and  $y \in [j/M - \xi, j/M + \xi]$  for some  $1 \le i, j \le M$ , we define

$$\begin{aligned} W_{s,\xi}(x, y) := & \left(\frac{1}{2} - \frac{x-i/M}{2\xi}\right) \left(\frac{1}{2} - \frac{y-j/M}{2\xi}\right) w_{i,j} \\ & + \left(\frac{1}{2} - \frac{x-i/M}{2\xi}\right) \frac{y-j/M+\xi}{2\xi} w_{i,j+1} \\ & + \frac{x-i/M+\xi}{2\xi} \left(\frac{1}{2} - \frac{y-j/M}{2\xi}\right) w_{i+1,j} \\ & + \frac{x-i/M+\xi}{2\xi} \cdot \frac{y-j/M+\xi}{2\xi} w_{i+1,j+1}. \end{aligned}$$

Note that  $W_{s,\xi}$  is Lipschitz continuous and that by construction we have  $\|W_{s,\xi} - W_s\|_{\square} \le 4M\xi \cdot \max_{i,j} w_{i,j}$  which approaches zero as  $\xi \to 0$ .

## 3 MODEL

**Finite Agent Model.** For the finite case, we assume that there are  $N$  agents with finite state and action spaces  $\mathcal{X}$  and  $\mathcal{U}$ , respectively. The agents implement actions at time points  $\mathcal{T} := \{0, \dots, T-1\}$  with terminal time point  $T$ . The interactions between individuals are modeled by a graph  $G_N = (V_N, E_N)$  where each vertex represents one agent and each edge a connection between two agents. For an arbitrary finite set  $A$  we denote by  $\mathcal{P}(A)$  the set of all probability measures on  $A$  and by  $\mathcal{B}(A)$  the set of all bounded measures on  $A$ . Thus, the space of policies is defined as  $\Pi := \mathcal{P}(\mathcal{U})^{\mathcal{T} \times \mathcal{X}}$  and a policy of agent  $i$  is denoted by  $\pi^i = (\pi_t^i)_{t \in \mathcal{T}} \in \Pi$  correspondingly. Furthermore, agents in the model are assumed to implement Markovian feedback policies such that they only consider local state information.

![Figure 1: Three networks and their empirical and mathematically expected degree distributions (DD). The figure shows three subplots: 'ER graph' (Erdős-Rényi), 'real Facebook graph', and 'power law graph'. Each subplot displays a network visualization at the top and a degree distribution plot at the bottom. The degree distribution plots show '# Nodes' on the y-axis (log scale from 10^0 to 10^2) and 'Degree' on the x-axis (linear scale from 0 to 100). Each plot compares 'expected DD' (red dashed line) and 'empirical DD' (black solid line). The ER graph shows a uniform distribution, the Facebook graph shows a power-law distribution with a few high-degree nodes, and the power law graph shows a similar power-law distribution.](e94f3bbb6f7501b9a1344dd0210e5dd8_img.jpg)

Figure 1: Three networks and their empirical and mathematically expected degree distributions (DD). The figure shows three subplots: 'ER graph' (Erdős-Rényi), 'real Facebook graph', and 'power law graph'. Each subplot displays a network visualization at the top and a degree distribution plot at the bottom. The degree distribution plots show '# Nodes' on the y-axis (log scale from 10^0 to 10^2) and 'Degree' on the x-axis (linear scale from 0 to 100). Each plot compares 'expected DD' (red dashed line) and 'empirical DD' (black solid line). The ER graph shows a uniform distribution, the Facebook graph shows a power-law distribution with a few high-degree nodes, and the power law graph shows a similar power-law distribution.

Figure 1: Three networks and their empirical and mathematically expected degree distributions (DD): Erdős-Rényi graph (left), real-world Facebook network (middle, data from Rozemberczki et al. (2019)), power law graph (right, for expected DD see Bollobás et al. (2007)); highly connected nodes are larger and darker: the Facebook graph has some nodes with high degrees and small and medium degrees otherwise. The power law graph generated by an  $L^p$  graphon is qualitatively similar. All nodes in the ER graph generated by a standard graphon have degrees smaller than thirty which contradicts the real-world network. Other standard graphons, e.g. ranked attachment (Borgs et al. (2011)), yield similar results as the ER graph but are omitted for space reasons. Each graph consists of 3892 nodes and around 17500 edges to match the real data set.

Formally, this is captured by defining for all  $t \in \mathcal{T}$  and  $i \in V_N$  the model dynamics

$$U_t^i \sim \pi_t^i(\cdot \mid X_t^i), \quad X_{t+1}^i \sim P(\cdot \mid X_t^i, U_t^i, \mathbb{G}_t^i) \quad (2)$$

with  $X_0^i \sim \mu_0$  for some transition kernel  $P: \mathcal{X} \times \mathcal{U} \times \mathcal{B}(\mathcal{X}) \to \mathcal{P}(\mathcal{X})$  and the neighborhood state distribution

$$\mathbb{G}_t^i := \frac{1}{N\rho_N} \sum_{j \in V_N} \mathbf{1}_{\{ij \in E_N\}} \delta_{X_t^j} \quad (3)$$

for each agent  $i$  with  $\delta$  being the Dirac measure and  $\mathbb{G}_t^i \in \mathcal{B}(\mathcal{X})$  for all  $i \le N$  by definition. The normalization factor  $\rho_N$  ensures that the neighborhood distribution does not converge to a vector of zeros as  $N$  approaches infinity. Here,  $\rho_N$  is assumed to have the same asymptotic order as the edge density of  $G_N$ , i.e.  $\rho_N = \Theta(|E_N|/N^2)$  as  $N \to \infty$ . Each agent faces a reward function  $r: \mathcal{X} \times \mathcal{U} \times \mathcal{B}(\mathcal{X}) \to \mathbb{R}$  which yields her reward depending on her state, action, and the state distribution of her neighbors. Agents competitively try to maximize their expected sum of rewards

$$J_i^N(\pi^1, \dots, \pi^N) := \mathbb{E} \left[ \sum_{t=0}^{T-1} r(X_t^i, U_t^i, \mathbb{G}_t^i) \right]. \quad (4)$$

Note that we can handle the infinite-horizon discounted objective case analogously, see e.g. Cui and Koeppl (2022). Finding equilibria for this type of model requires a suitable equilibrium concept. Although the classical Nash equilibrium notion (see, e.g. Carmona and Delarue (2018a)) seems to be a natural candidate, its definition requires that no agent has an incentive to unilaterally deviate from the current policy. As we are primarily interested in an approximation via  $L^p$  graphons, this equilibrium concept is too strict. Even in

the limit  $N \to \infty$  there can always occur (small) subgroups of agents whose graph connections deviate from the structure of the underlying graphon. Therefore, we work with the  $(\epsilon, p)$ -Markov-Nash equilibrium (see, for example Carmona (2004), Elie et al. (2020b), Cui and Koeppl (2022)) which only requires optimality for a fraction  $1-p$  of all individuals. This fraction will increase,  $(1-p) \to 1$  as  $N \to \infty$ .

**Definition 1.** An  $(\epsilon, p)$ -Markov-Nash equilibrium (MNE) for  $\epsilon, p > 0$  is defined as a tuple of policies  $\pi = (\pi^1, \dots, \pi^N) \in \Pi^N$  such that for any  $i \in \mathcal{W}_N$  we have

$$J_i^N(\pi) \ge \sup_{\pi \in \Pi} J_i^N(\pi^1, \dots, \pi^{i-1}, \pi, \pi^{i+1}, \dots, \pi^N) - \epsilon \quad (5)$$

for some  $\mathcal{W}_N \subseteq V_N$  with  $|\mathcal{W}_N| \ge \lfloor (1-p)N \rfloor$  such that  $\mathcal{W}_N$  contains at least  $\lfloor (1-p)N \rfloor$  agents.

**Mean Field Model.** The  $L^p$  graphon mean field model (LPGMFG) constitutes the limit of the finite agent model as  $N \to \infty$  and provides a reasonable approximation for the finite case. Before we formalize this claim and provide rigorous statements, we introduce the LPGMFG itself. The main difference to the  $N$ -agent model is that we now consider an infinite number of agents  $\alpha \in \mathcal{I} := [0, 1]$ . Thus,  $\mathcal{M}_t := \mathcal{P}(\mathcal{X})^{\mathcal{I}}$  denotes the space of measurable state marginal ensembles at time  $t$ , and  $\mathcal{M} := \mathcal{P}(\mathcal{X})^{\mathcal{I} \times \mathcal{T}}$  the space of measurable mean field ensembles. Here, measurable means that  $\alpha \mapsto \mu_t^\alpha(x)$  is measurable for all  $\mu \in \mathcal{M}, t \in \mathcal{T}, x \in \mathcal{X}$ . Analogously, a space of uniformly Lipschitz, measurable policy ensembles  $\Pi \subseteq \Pi^{\mathcal{I}}$  is defined such that  $\alpha \mapsto \pi_t^\alpha(u|x)$  is measurable and  $L_{\Pi}$ -Lipschitz for any  $\pi \in \Pi, t \in \mathcal{T}, u \in \mathcal{U}, x \in \mathcal{X}$ . Intuitively, a policy ensemble  $\pi \in \Pi$  includes an infinite number of policies

$\pi^\alpha \in \Pi$  where each policy is associated with one agent  $\alpha$ . State and action variables are defined for all  $(\alpha, t) \in \mathcal{I} \times \mathcal{T}$  as

$$U_t^\alpha \sim \pi_t^\alpha(\cdot | X_t^\alpha), \quad X_{t+1}^\alpha \sim P(\cdot | X_t^\alpha, U_t^\alpha, \mathbb{G}_t^\alpha), \quad (6)$$

with  $X_0^\alpha \sim \mu_0$  where the deterministic neighborhood MF of agent  $\alpha$  for some deterministic MF  $\mu \in \mathcal{M}$  is

$$\mathbb{G}_t^\alpha := \int_{\mathcal{I}} W(\alpha, \beta) \mu_t^\beta d\beta \quad (7)$$

with  $\mathbb{G}_t^\alpha \in \mathcal{B}(\mathcal{X})$  by definition. Each agent tries to competitively maximize her rewards given by

$$J_\alpha^\mu(\pi^\alpha) := \mathbb{E} \left[ \sum_{t=0}^{T-1} r(X_t^\alpha, U_t^\alpha, \mathbb{G}_t^\alpha) \right]. \quad (8)$$

Now, it remains to adapt the Nash equilibrium concept to the LPGMFG case. Thus, we introduce two functions  $\Psi: \Pi \to \mathcal{M}$  and  $\Phi: \mathcal{M} \to 2^\Pi$ .  $\Psi$  maps a policy ensemble  $\pi \in \Pi$  to the mean field ensemble  $\mu = \Psi(\pi) \in \mathcal{M}$  generated by  $\pi$  which is formalized by the recursive equation

$$\mu_{t+1}^\alpha(x) = \sum_{x' \in \mathcal{X}} \mu_t^\alpha(x') \pi_t^\alpha(u|x') P(x|x', u, \mathbb{G}_t^\alpha) \quad (9)$$

for all  $\alpha \in [0, 1]$  with  $\mu_0^\alpha \equiv \mu_0$ . The second map  $\Phi: \mathcal{M} \to 2^\Pi$  takes a mean field ensemble  $\mu \in \mathcal{M}$  and maps it to the set of policy ensembles  $\Phi(\mu) \subseteq 2^\Pi$  that are optimal with respect to  $\mu$ , i.e.  $\pi^\alpha = \arg \max_{\pi \in \Pi} J_\alpha^\mu(\pi^\alpha)$  for all  $\alpha \in [0, 1]$ . With the above definitions, we can state the equilibrium concept for LPGMFGs, namely the  $L^p$  graphon mean field equilibrium (GMFE).

**Definition 2.** A GMFE is a tuple  $(\mu, \pi) \in \Pi \times \mathcal{M}$  such that  $\pi \in \Phi(\mu)$  and  $\mu = \Psi(\pi)$ .

We also refer to the policy part of a GMFE as its Nash Equilibrium (NE). Intuitively, a GMFE consists of a policy ensemble  $\pi$  and a MF ensemble  $\mu$  such that  $\pi$  generates  $\mu$  and is also an optimal response to the generated MF. We will frequently use a Lipschitz assumption common in the MFG literature (Bayraktar et al. (2020), Carmona and Delarue (2018a), Cui and Koeppl (2022)) to enable the derivation of expressive theoretical results. The power law graphon, however, is not Lipschitz, so we derive a Lipschitz cutoff version in Appendix N. Since this cutoff power law graphon does not yield qualitatively different results compared to the power law graphon, it is omitted from the main text.

**Assumption 2.** Let  $r, P, W$  be Lipschitz continuous with Lipschitz constants  $L_r, L_P, L_W > 0$ , or alternatively there exist  $L_W > 0$  and disjoint intervals  $\{\mathcal{I}_1, \dots, \mathcal{I}_Q\}$ ,  $\cup_i \mathcal{I}_i = \mathcal{I}$  s.t.  $\forall i, j \le Q$  and  $\forall (x, y), (\tilde{x}, \tilde{y}) \in \mathcal{I}_i \times \mathcal{I}_j$ ,

$$|W(x, y) - W(\tilde{x}, \tilde{y})| \le L_W(|x - \tilde{x}| + |y - \tilde{y}|). \quad (10)$$

Under Assumption 2, the model defined above has a GMFE which is formalized by the next theorem.

**Theorem 1.** Under Assumption 2 and for Lipschitz  $W$ , there exists a GMFE  $(\pi, \mu) \in \Pi \times \mathcal{M}$ .

*Proof.* The existence of a GMFE follows from (Saldi et al., 2018, Theorem 3.3) for the extended state space  $\mathcal{X} \times [0, 1]$ . See also (Cui and Koeppl, 2022, Proof of Theorem 1).  $\square$

**Mean Field Approximation.** The proofs of all theoretical results can be found in the Appendix. This paragraph relates the finite agent model to the MF model by showing that LPGMFGs yield an increasingly accurate approximation for the  $N$ -agent case as the number of agents grows. We emphasize that the LPGMFG yields an approximation for the  $N$ -agent game for all  $N$  at once. Both the theoretical results as well as the empirical findings show that the accuracy of this approximation increases with the number  $N$  of agents, see the next sections for details. As a consequence, the LPGMFG concept provides a scalable and increasingly accurate approximation for otherwise intractable multi agent problems with a large number of individuals. By Theorem 1, there exists a GMFE  $(\pi, \mu)$  which yields an approximate NE for the  $N$ -agent problem through the map  $\Gamma_N(\pi) := (\pi^1, \dots, \pi^N) \in \Pi^N$  defined by  $\pi_t^i(u|x) := \pi_t^{\alpha_i}(u|x)$  for all  $\alpha \in \mathcal{I}$ ,  $t \in \mathcal{T}$ ,  $x \in \mathcal{X}$ ,  $u \in \mathcal{U}$  with  $\alpha_i = i/N$ .

For a theoretical comparison, we lift both the policies and empirical distributions in the finite agent model to the continuous domain  $\mathcal{I}$ . Thus, for an  $N$ -agent policy tuple  $(\pi^1, \dots, \pi^N) \in \Pi^N$  the corresponding step policy ensemble  $\pi^N \in \Pi$  and the random empirical step measure ensemble  $\mu^N \in \mathcal{M}$  are defined by  $\pi_t^{N,\alpha} := \sum_{i \in V_N} \mathbf{1}_{\{\alpha \in (\frac{i-1}{N}, \frac{i}{N}]\}} \cdot \pi_t^i$  and  $\mu_t^{N,\alpha} := \sum_{i \in V_N} \mathbf{1}_{\{\alpha \in (\frac{i-1}{N}, \frac{i}{N}]\}} \cdot \delta_{X_t^i}$  for all  $\alpha \in \mathcal{I}$  and  $t \in \mathcal{T}$ . For notational convenience, we furthermore define for any  $f: \mathcal{X} \times \mathcal{I} \to \mathbb{R}$  and state marginal ensemble  $\mu_t \in \mathcal{M}_t$ ,  $\mu_t(f) := \int_{\mathcal{I}} \sum_{x \in \mathcal{X}} f(x, a) \mu_t^\alpha(x) d\alpha$ . With these definitions in place, we state our first main theoretical result.

**Theorem 2.** Consider  $\pi \in \Pi$  with  $\mu = \Psi(\pi)$ . Under Assumption 1 and the  $N$ -agent policy  $(\pi^1, \dots, \pi^{i-1}, \hat{\pi}, \pi^{i+1}, \dots, \pi^N) \in \Pi^N$  with  $(\pi^1, \pi^2, \dots, \pi^N) = \Gamma_N(\pi) \in \Pi^N$ ,  $\hat{\pi} \in \Pi$ ,  $t \in \mathcal{T}$ , we have for all measurable functions  $f: \mathcal{X} \times \mathcal{I} \to \mathbb{R}$  uniformly bounded by some  $M_f > 0$ , that

$$\mathbb{E} [| \mu_t^N(f) - \mu_t(f) |] \to 0 \quad (11)$$

uniformly over all possible deviations  $\hat{\pi} \in \Pi$ ,  $i \in V_N$ . If the graphon convergence in Assumption 1 is up to rate  $O(1/\sqrt{N})$ , then this rate of convergence is the same.

Based on Theorem 2, we can derive a central result of this paper which formalizes the capability of LPGMFGs to approximate finite  $N$ -agent models. In contrast to prior work, proving Theorem 2 requires an additional mathematical effort which is discussed in Appendix A.

**Theorem 3.** Consider a GMFE  $(\pi, \mu)$  under Assumptions 1 and 2. For any  $\varepsilon, p > 0$  there exists  $N'$  such that for all  $N > N'$ , the policy  $\Gamma_N(\pi) \in \Pi^N$  is an  $(\varepsilon, p)$ -MNE.

Intuitively, Theorem 3 states that the GMFE provides an increasingly accurate approximation of the  $N$ -agent problem as the number of agents goes up. Since the algorithmic computation of NE is in general intractable (Conitzer and Sandholm (2008), Papadimitriou (2001), Papadimitriou (2007)), the LPGMFGs approximation can overcome these difficulties by choosing  $\varepsilon$  and  $p$  in Theorem 3 close to zero when the number  $N$  of agents is sufficiently large.

## 4 LEARNING LPGMFGS

**Equivalence Class Method.** For learning equilibria in LPGMFGs, we introduce equivalence classes (Cui and Koeppel (2022)). We discretize the continuous interval  $\mathcal{I}$  of agents by some finite number  $M$  of subintervals that form a partition of  $\mathcal{I}$ . For convenience, we usually assume that every subinterval has the same length. Then, all agents within one class, i.e. a subinterval, are approximated by the agent who is located at the center of the respective subinterval. Subsequently, we can solve the optimal control problem for each equivalence class separately by applying either backwards induction or RL. Although this formulation seems to resemble classical multi-population mean field games (MP MFGs) (Huang et al. (2006), Perolat et al. (2021)) at first, the crucial advantages of LPGMFGs are that they are on the one hand rigorously connected to finite agent games. On the other hand, they can handle an uncountable number of agent equivalence classes that cannot be captured by the standard multi-class model. Beyond that, the just described learning method for LPGMFGs does not just provide an approximation for some finite  $N$ -agent problem with a fixed  $N$ . Instead, it yields an estimation for the  $N$ -agent problem for all arbitrary, large enough  $N$  at once. The technical details of the approach can be found in Appendix I.

**Online Mirror Descent (OMD).** The discretized game generated by the equivalence class method can be interpreted as a MP MFG with  $M$  populations. In the literature, the concept of OMD is used to learn equilibria in such MP MFGs (Hadikhanloo (2017), Perolat et al. (2021)). Our paper leverages these concepts to learn LPGMFGs.

To prove convergence for the OMD algorithm, we have to ensure that a NE exists. Here, the discretized GMFG can be interpreted as a GMFG on the step-graphon  $W_s$  created by discretization. To facilitate the theoretical analysis of the OMD algorithm, we consider the corresponding smoothed step graphon  $W_{s,\xi}$  and the smoothed GMFG given by the dynamics  $\hat{U}_t^\alpha \sim \pi_t^\alpha(\cdot | \hat{X}_t^\alpha)$  and  $\hat{X}_{t+1}^\alpha \sim P(\cdot | \hat{X}_t^\alpha, \hat{U}_t^\alpha, \hat{\mathbb{G}}_t^\alpha)$ , with  $\hat{X}_0^\alpha \sim \mu_0$  for all  $(\alpha, t) \in \mathcal{I} \times \mathcal{T}$  where  $\hat{\mathbb{G}}$  is the neighborhood state distribution for the smoothed step graphon. One advantage of this approach is that the existence of a

GMFE  $(\mu_{s,\xi}, \pi_{s,\xi})$  is ensured by Theorem 1 for this GMFG. Also, for  $\xi$  close enough to zero, the smoothed step graphon converges to the original step graphon in the cut norm.

**Theorem 4.** Suppose that  $(\mu_{s,\xi}, \pi_{s,\xi})$  is a GMFE in the smoothed version of the MP MFG on the step graphon  $W$  under Assumption 2. Then, for every  $\varepsilon, p > 0$  there exists a  $\xi' > 0$  such that for all  $0 < \xi < \xi'$

$$\sup_{\pi \in \Pi} J_{\alpha,W}(\pi_{s,\xi}) - J_{\alpha,W}(\pi) \le \varepsilon \quad (12)$$

for all  $\alpha \in \mathcal{J}$  for some  $\mathcal{J} \subseteq \mathcal{I}$  with Lebesgue measure  $\lambda(\mathcal{J}) \ge 1 - p$ .

This means that a GMFE in the smoothed version of the GMFG is an  $(\varepsilon, p)$ -MNE for the discretized game. Combining this insight with existing results (Cui and Koeppel, 2022, Theorem 5) indicates that the smoothed GMFE provides a good approximation for the finite agent case, but we leave a rigorous proof for future work. We call a smoothed MP MFG weakly monotone if for any  $\pi, \pi' \in \Pi$  we have

$$\begin{aligned} \tilde{d}(\pi, \pi') := \int_{\mathcal{I}} \left[ J_{\alpha}^{\mu}(\pi^{\alpha}) + J_{\alpha}^{\mu'}(\pi'^{\alpha}) \right. \\ \left. - J_{\alpha}^{\mu}(\pi'^{\alpha}) - J_{\alpha}^{\mu'}(\pi^{\alpha}) \right] d\alpha \le 0 \end{aligned} \quad (13)$$

where  $\mu = \Psi(\pi)$  and  $\mu' = \Psi(\pi')$  are the MFs associated with the respective policies. If the inequality is strict  $\forall \pi \neq \pi'$ , we call the MP MFG strictly weakly monotone.

**Assumption 3.** The smoothed MP MFG is strictly weakly monotone.

Weak monotonicity can be interpreted as agents preferring less crowded areas over crowded ones. Under Assumption 3, the NE guaranteed by Theorem 1 is unique.

**Lemma 1.** If the smoothed MP MFG satisfies Assumptions 2 and 3, it has a unique NE.

We define the OMD algorithm as in Perolat et al. (2021) and consider the continuous time case (CTOMD) where we denote the time of the algorithm by  $\tau > 0$ . Then, we obtain the following convergence result.

**Theorem 5.** If the smoothed MP MFG satisfies Assumptions 2 and 3 and the transition kernel does not depend on the MF, the sequence of policies  $(\pi_{\tau})_{\tau \ge 0}$  generated by the CTOMD algorithm converges to the unique NE as  $\tau \to \infty$ .

Empirical evidence collected in our simulations suggests that a convergence guarantee as in Theorem 5 also holds for the case where the MF depends on the transition kernel, but we leave a rigorous proof for future work.

## 5 EXPERIMENTS

In our experiments, we use OMD with its hyperparameter  $\gamma$  set to 1, the power law  $L^p$ -graphon  $W$ , and discretize  $\mathcal{I}$

![Figure 2: Experimental results for OMD on the Cyber Security problem. (a) A line plot of exploitability ΔJ vs iteration n, showing a decreasing trend from 10^1 to 10^-2. (b)-(e) Heatmaps of graphon index α vs time t for states DI, DS, UI, and US respectively, showing the probability π_t^α(1|DI), π_t^α(1|DS), π_t^α(1|UI), and π_t^α(1|US). (f) A line plot of infection probability μ_t^α(I) vs time t for discretized α, showing multiple curves that converge to a stable value around 0.5.](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

Figure 2: Experimental results for OMD on the Cyber Security problem. (a) A line plot of exploitability ΔJ vs iteration n, showing a decreasing trend from 10^1 to 10^-2. (b)-(e) Heatmaps of graphon index α vs time t for states DI, DS, UI, and US respectively, showing the probability π\_t^α(1|DI), π\_t^α(1|DS), π\_t^α(1|UI), and π\_t^α(1|US). (f) A line plot of infection probability μ\_t^α(I) vs time t for discretized α, showing multiple curves that converge to a stable value around 0.5.

Figure 2: Experimental results for OMD on the Cyber Security problem. (a): The exploitability  $\Delta J$  over iterations  $n$  of OMD; (b)-(e): The probability of choosing action  $u = 1$  at graphon index  $\alpha$  and time  $t$  under the final equilibrium policy in states  $DI, DS, UI, US$  respectively; (f): The probability (mean-field) of infected agents, visualized for each discretized  $\alpha$ .

into  $M = 25$  subintervals for the Cyber Security problem or  $M = 10$  for the Beach Bar problem given as follows. Here, we emphasize that using  $L^p$ -graphons in the experiments is a key component of our LPFGMFG approach. This allows us to model many realistic networks which are characterized by sparsity and power law degree distributions. As we discussed previously, standard GMFG approaches are conceptually unable to capture these networks.

**Cyber Security.** We modify an existing cyber security model (Carmona and Delarue (2018a), Kolokoltsov and Bensoussan (2016)) where a virus spreads to computers either directly by an attack, or by other nearby infected computers. In contrast to existing work, we use LPGMFGs to allow malware spread only by neighboring computers to increase the modelling accuracy. Each computer can be either infected ( $I$ ) or susceptible ( $S$ ), as well as defended ( $D$ ) or unprotected ( $U$ ), formally  $\mathcal{X} := \{DI, DS, UI, US\}$ . Agents may attempt to switch (with geometrically distributed delay) between defense states,  $\mathcal{U} := \{0, 1\}$ . The recovery and infection probabilities depend on the defense state and number of infected neighbors, while the reward function consists of costs for being defended or infected. Details can be found in Appendix O.1.

**Heterogeneous Cyber Security.** A natural extension of the cyber security model is the adaptation to a multi-class framework with heterogeneous agent classes. For illustrative purposes we focus on only two types of agents – private (Pri) and corporate (Cor), see Appendix O.2 for details.

**Beach Bar Process.** Introduced as the Santa Fe bar problem (Arthur (1994), Farago et al. (2002)), variations of the

Beach Bar Process are frequently used to demonstrate the capabilities of learning algorithms (Perolat et al. (2021), Perrin et al. (2020)). Agents can move their towels between locations and try to be close to the bar but also avoid crowded areas and neighbors in an underlying network. Formally, we consider a one-dimensional beach bar process with  $|\mathcal{X}| = 10$  locations  $\mathcal{X} = \{0, 1, \dots, |\mathcal{X}| - 1\}$  where a bar is located in the middle  $B = \lfloor |\mathcal{X}|/2 \rfloor$  of the beach. The  $N$  agents may move their towel between locations,  $\mathcal{U} = \{-1, 0, 1\}$ . Furthermore, the agents are connected by a power law network where connected agents try to avoid being close to each other. See Appendix O.3 for details.

**Experimental Results.** As seen in Figures 2 and 3, the approximate exploitability  $\Delta J(\pi) = \int_{\mathcal{T}} \sup_{\pi^* \in \Pi} J_{\alpha}^{\Psi(\pi)}(\pi^*) - J_{\alpha}^{\Psi(\pi)}(\pi) d\alpha$  of a MF policy  $\pi$  quantifies the sub-optimality of the obtained equilibrium and quickly converges in the Cyber Security and Beach Bar scenario using OMD. We obtain near-identical results also for the Heterogeneous Cyber Security problem, which are omitted for space reasons. The algorithm converges to an equilibrium where, as expected, the agents with the most connections in the graph attempt to defend at fixed cost, as their expected cost from not defending is higher than for agents with fewer connections. The system quickly runs into an almost time-stationary state, where the costs of defending equilibrate with the expected cost of future infection, see Figure 2. Since we consider a finite-horizon however, the option of defending becomes increasingly unattractive as time runs out. The probability of an agent  $\alpha$  being infected at any time shows an interesting behavior: At  $\alpha = 0$ , the probability is quite high due to the great number of connections. As  $\alpha$  decreases, so does the probability of

![Figure 3: Experimental results for OMD on the Beach problem. (a) A line plot of exploitability ΔJ vs iteration n on a log scale. (b) A heatmap of the final distribution over positions x and graphon index α. (c) A 3D surface plot of the final mean-field μ_t^α(x) over time t and graphon index α.](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

Figure 3 consists of three subplots. (a) is a line plot showing the exploitability  $\Delta J$  on a logarithmic y-axis (from  $10^{-2}$  to  $10^1$ ) against the iteration number  $n$  (from 0 to 400). The exploitability decreases from approximately 10 to  $10^{-2}$ . (b) is a heatmap showing the final distribution over positions  $x$  (x-axis, 0 to 10) and graphon index  $\alpha$  (y-axis, 0 to 1). The color scale represents the probability density, ranging from 0.0 to 0.6. (c) is a 3D surface plot showing the final mean-field  $\mu_t^{\alpha}(x)$  as a function of time  $t$  (x-axis, 0 to 50) and graphon index  $\alpha$  (y-axis, 0 to 1). The color scale represents the mean-field value, ranging from 0.00 to 0.25.

Figure 3: Experimental results for OMD on the Beach problem. (a) A line plot of exploitability ΔJ vs iteration n on a log scale. (b) A heatmap of the final distribution over positions x and graphon index α. (c) A 3D surface plot of the final mean-field μ\_t^α(x) over time t and graphon index α.

Figure 3: Experimental results for OMD on the Beach problem. (a) The exploitability  $\Delta J$  over iterations  $n$  of OMD; (b) The final distribution over positions on the beach at time  $t = T - 1$  for each discretized  $\alpha$ ; (c) The evolution of distributions over time.

![Figure 4: Experimental results for OMD on the heterogeneous Cyber Security problem. (a)-(d) Heatmaps of action probability u=1 over time t and graphon index α for states CorDI, CorDS, CorUI, CorUS. (e) Heatmap of infection probability for Cor agents. (f)-(j) Heatmaps of action probability and infection probability for Pri agents.](d864789b0d8384da1d22fd6a5d76bbdf_img.jpg)

Figure 4 consists of ten subplots arranged in two rows. The top row (a-e) shows results for Cor agents, and the bottom row (f-j) shows results for Pri agents. (a), (c), (e), (g), and (i) are heatmaps of the probability of action  $u = 1$  over time  $t$  (0 to 25) and graphon index  $\alpha$  (0 to 1) for states CorDI, CorDS, CorUI, CorUS, and PriDS, respectively. (b), (d), (f), (h), and (j) are heatmaps of the infection probability  $\mu_t^{\alpha}(I | \text{Cor})$  or  $\mu_t^{\alpha}(I | \text{Pri})$  over the same axes. The color scale for the action probability heatmaps ranges from 0.0 to 1.0, and for the infection probability heatmaps, it ranges from 0.0 to 1.0.

Figure 4: Experimental results for OMD on the heterogeneous Cyber Security problem. (a)-(d) Heatmaps of action probability u=1 over time t and graphon index α for states CorDI, CorDS, CorUI, CorUS. (e) Heatmap of infection probability for Cor agents. (f)-(j) Heatmaps of action probability and infection probability for Pri agents.

Figure 4: Experimental results for OMD on the heterogeneous Cyber Security problem. (a)-(d): The probability of action  $u = 1$  at graphon index  $\alpha$  and time  $t$  under the final equilibrium policy in states  $\text{CorDI}$ ,  $\text{CorDS}$ ,  $\text{CorUI}$ ,  $\text{CorUS}$ ; (e): The probability (MF) of infected Cor agents, visualized for each discretized  $\alpha$ ; (f)-(j): Same as in (a)-(e) but for Pri agents.

being infected at all times. However, as soon as  $\alpha$  passes a threshold where defense is no longer individually worth it, the fraction of infected nodes jumps up.

In the heterogeneous case, as seen in Figure 4, we consider an additional class of nodes with partially similar behavior. For very high connectivity  $\alpha \to 0$  however, we observe that Pri nodes will never defend themselves, since for the given problem parameters, the probability of infection will be very high regardless of the defense state. Otherwise, we can observe similar behavior as in the homogeneous case. Perhaps most interesting is the asymmetry between choosing to switch between defended and undefended. When agents are susceptible, some agents will opt to neither switch from defended to undefended, nor vice versa. This stems from the fact that agents switching in state  $US$  could likely jump to  $UI$  and  $DI$ , while in state  $DS$  likely jumps are  $DS$  and  $US$ , each of which may have different future returns. For the Beach Bar process in Figure 3, we see results similar to the classical ones in Perrin et al. (2020). By giving each agent

![Figure 5: A line plot showing the L1 error between the empirical distribution and the limiting MF Δμ vs the number of agents N. Two lines are shown: Cyber (red solid) and Beach (blue dashed).](aa14b9ec884bf40ce06c161be468cd84_img.jpg)

Figure 5 is a line plot showing the mean-field deviation  $\Delta \mu$  on the y-axis (0 to 30) against the number of agents  $N$  on the x-axis (0 to 80). Two lines are plotted: 'Cyber' (red solid line) and 'Beach' (blue dashed line). Both lines show a decreasing trend as  $N$  increases, starting from a high value (around 30) at  $N=0$  and approaching a steady-state value (around 10) as  $N$  reaches 80. Shaded regions around the lines indicate the 68% confidence interval.

Figure 5: A line plot showing the L1 error between the empirical distribution and the limiting MF Δμ vs the number of agents N. Two lines are shown: Cyber (red solid) and Beach (blue dashed).

Figure 5: The  $L_1$  error between the empirical distribution and the limiting MF  $\Delta \mu = \mathbb{E} \left[ \sum_{t \in \mathcal{T}, x \in \mathcal{X}} \left| \frac{1}{N} \sum_i \delta_{X_t^i}(x) - \int_{\mathcal{X}} \mu_t^{\alpha}(x) d\alpha \right| \right]$  at  $\beta = 0.51$  averaged over 100 randomly sampled graphs with  $N$  nodes and 68% confidence interval (shaded).

![Figure 6: A heatmap showing the L1 error (Deviation Δμ) between the empirical distribution and the limiting MF for the Cyber Security problem. The x-axis represents the Number of agents N (0 to 100) and the y-axis represents the Sparsity parameter β (0.0 to 1.0). The color scale ranges from 10 (dark) to 50 (light). The error is highest (lightest colors) for small N and β near 0 or 1, and decreases (darker colors) as N increases and β moves towards 0.5.](b93cbfb52e37619e688175a6aad9edd9_img.jpg)

Figure 6: A heatmap showing the L1 error (Deviation Δμ) between the empirical distribution and the limiting MF for the Cyber Security problem. The x-axis represents the Number of agents N (0 to 100) and the y-axis represents the Sparsity parameter β (0.0 to 1.0). The color scale ranges from 10 (dark) to 50 (light). The error is highest (lightest colors) for small N and β near 0 or 1, and decreases (darker colors) as N increases and β moves towards 0.5.

Figure 6: The  $L_1$  error between the empirical distribution and the limiting MF as in Figure 5 over 50 uniformly spaced  $\beta \in (0, 1)$  and  $N \le 100$  for the Cyber Security problem.

an incentive to avoid only their direct graphical neighbors, we obtain an equilibrium behavior where agents with many connections will stay further away from the bar, while agents with few connections will not mind many other agents.

Finally, in Figures 5 and 6 for  $\rho_N = N^{-\beta}$  and sparsity parameter  $\beta \in (0, 1)$ , we observe convergence of the  $N$ -agent system objective to the MF objective, implying that sufficiently large finite systems are well-approximated by the LPGMFG. In Figure 6, for  $\beta$  close to 0 or 1, convergence slows down, as by (Borgs et al., 2019, Theorem 2.14) convergence is only guaranteed for  $0 < \beta < 1$ . Even though for  $\beta = 0$ , one would get the same model as in Cui and Koeppl (2022), since the power law graphon is not  $[0, 1]$ -valued, approximation guarantees fail for  $\beta = 0$  and we observe increasingly slow convergence as we approach zero.

## 6 CONCLUSION

In this paper we have introduced LPGMFGs which enable the scalable, mathematically sound analysis of otherwise intractable MARL problems on large sparse graphs. We rigorously derived existence and convergence guarantees for LPGMFGs and provided learning schemes to find equilibria algorithmically where we adapted the OMD learning algorithm to the setting of LPGMFGs. Beyond that, we demonstrated the benefits of our approach empirically on different examples and showed that the practical results match the theory. As for the societal impact we foresee from our work, we believe that while our techniques remain very general, they could in the future lead to an analysis of self-interested agents on real graphs such as from social networks. This could find application e.g. in control strategies for future pandemics, or other interventions. Future work could extend our model in numerous ways such as considering continuous time, action, and state spaces or adding noise terms. A challenging task could also be to find similar learning concepts for ultra-sparse graphs where the degrees remain constant as the number of agents becomes large. For applications, it would be interesting to use LPGMFGs to solve real-world problems that occur in various research

fields. In general, we hope that our work contributes to the MARL literature and inspires future work on scalable learning methods on sparse graphs.

## Acknowledgements

This work has been co-funded by the Hessian Ministry of Science and the Arts (HMWK) within the projects "The Third Wave of Artificial Intelligence - 3AI" and hessian.AI, and the LOEWE initiative (Hesse, Germany) within the emergenCITY center.

## References

- Anahtarcı, B., Kariksz, C. D., and Saldi, N. (2020). Q-learning in regularized mean-field games. *arXiv preprint arXiv:2003.12151*.
- Aparicio, S., Villazón-Terrazas, J., and Álvarez, G. (2015). A model for scale-free networks: application to twitter. *Entropy*, 17(8):5848–5867.
- Arthur, W. B. (1994). Inductive reasoning and bounded rationality. *The American Economic Review*, 84(2):406–411.
- Aurell, A., Carmona, R., Dayanikli, G., and Laurière, M. (2022). Finite state graphon games with applications to epidemics. *Dynamic Games and Applications*, pages 1–33.
- Aurell, A., Carmona, R., and Laurière, M. (2021). Stochastic graphon games: II. the linear-quadratic case. *arXiv preprint arXiv:2105.12320*.
- Avena-Koenigsberger, A., Misic, B., and Sporns, O. (2018). Communication dynamics in complex brain networks. *Nature Reviews Neuroscience*, 19(1):17–33.
- Bakker, L., Hare, W., Khosravi, H., and Ramadanovic, B. (2010). A social network model of investment behaviour in the stock market. *Physica A: Statistical Mechanics and its Applications*, 389(6):1223–1229.
- Barabási, A.-L. and Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439):509–512.
- Barabási, A.-L., Albert, R., and Jeong, H. (1999). Mean-field theory for scale-free random networks. *Physica A: Statistical Mechanics and its Applications*, 272(1-2):173–187.
- Bayraktar, E., Chakraborty, S., and Wu, R. (2020). Graphon mean field systems. *arXiv preprint arXiv:2003.13180*.
- Bertsekas, D. (2019). *Reinforcement learning and optimal control*. Athena Scientific.
- Bian, Y.-t., Xu, L., and Li, J.-s. (2016). Evolving dynamics of trading behavior based on coordination game in complex networks. *Physica A: Statistical Mechanics and its Applications*, 449:281–290.

- Bollobás, B., Janson, S., and Riordan, O. (2007). The phase transition in inhomogeneous random graphs. *Random Structures & Algorithms*, 31(1):3–122.
- Borgs, C., Chayes, J., Cohn, H., and Zhao, Y. (2018). An Lp theory of sparse graph convergence II: Ld convergence, quotients and right convergence. *The Annals of Probability*, 46(1):337–396.
- Borgs, C., Chayes, J., Cohn, H., and Zhao, Y. (2019). An Lp theory of sparse graph convergence I: Limits, sparse random graph models, and power law distributions. *Transactions of the American Mathematical Society*, 372(5):3019–3062.
- Borgs, C., Chayes, J., Lovász, L., Sós, V., and Vesztergombi, K. (2011). Limits of randomly grown graph sequences. *European Journal of Combinatorics*, 32(7):985–999.
- Bullmore, E. and Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3):186–198.
- Bullmore, E. and Sporns, O. (2012). The economy of brain network organization. *Nature Reviews Neuroscience*, 13(5):336–349.
- Caines, P. E. and Huang, M. (2019). Graphon mean field games and the GMFG equations:  $\varepsilon$ -Nash equilibria. In *2019 IEEE 58th Conference on Decision and Control (CDC)*, pages 286–292. IEEE.
- Canese, L., Cardarilli, G. C., Di Nunzio, L., Fazzolari, R., Giardino, D., Re, M., and Spanò, S. (2021). Multi-agent reinforcement learning: A review of challenges and applications. *Applied Sciences*, 11(11):4948.
- Cardaliaguet, P. and Hadikhanloo, S. (2017). Learning in mean field games: the fictitious play. *ESAIM: Control, Optimisation and Calculus of Variations*, 23(2):569–591.
- Carmona, G. (2004). Nash equilibria of games with a continuum of players.
- Carmona, R., Cooney, D. B., Graves, C. V., and Lauriere, M. (2022). Stochastic graphon games: I. the static case. *Mathematics of Operations Research*, 47(1):750–778.
- Carmona, R. and Delarue, F. (2018a). *Probabilistic Theory of Mean Field Games with Applications I: Mean Field FBSDEs, Control, and Games*, volume 83. Springer.
- Carmona, R. and Delarue, F. (2018b). *Probabilistic Theory of Mean Field Games with Applications II: Mean Field Games with Common Noise and Master Equations*. Probability Theory and Stochastic Modelling. Springer International Publishing.
- Carmona, R., Fouque, J.-P., and Sun, L.-H. (2015). Mean field games and systemic risk. *Communications in Mathematical Sciences*, 13(4):911–933.
- Carmona, R., Laurière, M., and Tan, Z. (2019). Model-free mean-field reinforcement learning: mean-field mdp and mean-field q-learning. *arXiv preprint arXiv:1910.12802*.
- Carmona, R. and Zhu, X. (2016). A probabilistic approach to mean field games with major and minor players. *The Annals of Applied Probability*, 26(3):1535–1580.
- Castiglione, A., Gribaudo, M., Iacono, M., and Palmieri, F. (2014). Exploiting mean field analysis to model performances of big data architectures. *Future Generation Computer Systems*, 37:203–211.
- Choutri, S. E. and Djehiche, B. (2019). Mean-field risk sensitive control and zero-sum games for markov chains. *Bulletin des Sciences Mathématiques*, 152:1–39.
- Choutri, S. E., Djehiche, B., and Tembine, H. (2019). Optimal control and zero-sum games for markov chains of mean-field type. *Mathematical Control & Related Fields*, 9(3):571.
- Concas, G., Marchesi, M., Pinna, S., and Serra, N. (2007). Power-laws in a large object-oriented software system. *IEEE Transactions on Software Engineering*, 33(10):687–708.
- Conitzer, V. and Sandholm, T. (2008). New complexity results about Nash equilibria. *Games and Economic Behavior*, 63(2):621–641.
- Cui, K. and Koeppl, H. (2021). Approximately solving mean field games via entropy-regularized deep reinforcement learning. In *International Conference on Artificial Intelligence and Statistics*, pages 1909–1917. PMLR.
- Cui, K. and Koeppl, H. (2022). Learning graphon mean field games and approximate Nash equilibria. In *International Conference on Learning Representations*.
- Cui, K., Li, M., Fabian, C., and Koeppl, H. (2022). Scalable task-driven robotic swarm control via collision avoidance and learning mean-field control. *arXiv preprint arXiv:2209.07420*.
- Delarue, F. and Vasileiadis, A. (2021). Exploration noise for learning linear-quadratic mean field games. *arXiv preprint arXiv:2107.00839*.
- D’Arcangelis, A. M. and Rotundo, G. (2016). Complex networks in finance. In *Complex Networks and Dynamics*, pages 209–235. Springer.
- Elamvazhuthi, K. and Berman, S. (2019). Mean-field models in swarm robotics: A survey. *Bioinspiration & Biomimetics*, 15(1):015001.
- Elie, R., Ichiba, T., and Laurière, M. (2020a). Large banking systems with default and recovery: A mean field game model. *arXiv preprint arXiv:2001.10206*.
- Elie, R., Perolat, J., Laurière, M., Geist, M., and Pietquin, O. (2020b). On the convergence of model free learning in mean field games. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 7143–7150.
- Farago, J., Greenwald, A., and Hall, K. (2002). Fair and efficient solutions to the santa fe bar problem. In *Grace Hopper Celebration of Women in Computing*. Citeseer.