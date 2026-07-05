# Literature And Engine-Design Survey

## Classic game-engine evaluation and tuning

- Claude Shannon, "Programming a Computer for Playing Chess" (1949). Foundational additive evaluation model.
- Chessprogramming Wiki, `Evaluation`: practical survey of material, mobility, king safety, phase, tempo, and symmetry.
- Stockfish classical eval / Stockfish Evaluation Guide: production example of interpretable terms with debug output.
- Fishtest: production evidence that paired H2H and statistical gating are the source of truth.
- Peter Oesterlund, "Texel's Tuning Method": logistic fitting of eval values to game outcomes; best first choice for many static weights.
- Michael Buro, "Statistical Feature Combination for the Evaluation of Game Positions" (JAIR 1995): peer-reviewed support for logistic regression over handcrafted features.
- SPSA (Spall 1998; Chessprogramming SPSA): high-value tuning method for noisy H2H, because each step uses two perturbations independent of parameter dimension.
- CLOP (Coulom): useful for low-dimensional noisy parameters.
- CMA-ES (Hansen, arXiv:1604.00772): useful for small nonlinear parameter blocks; too sample-hungry for many weights.
- TDLeaf(lambda) (Baxter, Tridgell, Weaver, arXiv:cs/9901001): search-integrated temporal-difference tuning; powerful but riskier under Pokemon's high stochasticity.

Implications:

- Build the eval as feature groups with tunable weights.
- Fit initial weights offline with logistic/Texel-style tuning.
- Promote only via paired H2H and later ladder GXE.
- Use SPSA for search-coupled knobs after static fitting.

## Imperfect information and uncertainty

- Frank & Basin, "Search in games with incomplete information" (AIJ 1998): strategy fusion and non-locality. Determinized search assumes it can act differently in different hidden worlds.
- Long, Sturtevant, Buro, Furtak, "Understanding the Success of Perfect Information Monte Carlo Sampling" (AAAI 2010): PIMC works best with high leaf correlation and high disambiguation. Randbats fits this better than poker, but residual EV lives in information/scouting/concealment.
- Cowling, Powley, Whitehouse, "Information Set Monte Carlo Tree Search" (IEEE TCIAIG 2012): information-set statistics reduce but do not eliminate determinization pathologies.
- Silver & Veness, "Monte-Carlo Planning in Large POMDPs" (NeurIPS 2010): particle beliefs are a practical approximation for large hidden-state planning.
- ReBeL (Brown et al., arXiv:2007.13544) and Student of Games (Schmid et al., arXiv:2112.03178): correct theoretical object is the public belief state, but exact belief-state search is too expensive early-game.
- Nashed & Zilberstein, "A Survey of Opponent Modeling in Adversarial Domains" (JAIR 2022): opponent behavior should update beliefs, not only reveal consistency.

Implications:

- Do not only sample one concrete set and pretend it's known.
- Carry belief summaries into eval.
- Add threat/scout/concealment terms as cheap approximations to belief-state search.
- Keep these terms bounded; they should bias search, not override tactics.

## Simultaneous-move search

- Lisý et al., "Convergence of Monte Carlo Tree Search in Simultaneous Move Games" (NeurIPS 2013, arXiv:1310.8613): Exp3 / regret-matching can converge in simultaneous move games under assumptions.
- Kovařík & Lisý, "Analysis of Hannan consistent selection..." (Machine Learning 2020): guarantees are subtle; average strategies matter.
- Saffidine, Finnsson, Buro, "Alpha-Beta Pruning for Games with Simultaneous Moves" (AAAI 2012): pruning and matrix solving can help once action spaces are small.

Implications:

- DUCT is theoretically unsound but empirically strong for FP.
- Do not spend major effort replacing DUCT before higher-value eval gaps are exhausted.
- Late-game exact solving / matrix re-solving is the plausible sound-search upgrade.

## Pokemon-specific sources

- `pmariglia/foul-play` and `pmariglia/poke-engine`: strongest public search baseline; deterministic MCTS + set inference + fast Rust sim.
- Foul Play Smogon thread and pmariglia blog: maintainer confirms long-horizon, stall/Toxic, and eval tuning limitations; reports strong gen9randombattle ladder results at higher search budgets.
- Pokemon Showdown `data/random-battles/gen9/sets.json` and `teams.ts`: authoritative randbats generator; exact public prior.
- `pkmn/randbats`: generated randbats set distributions.
- Oak / "Stockfish for RBY" Smogon thread and repo: fast-engine + search + small eval/policy nets; emphasizes Exp3 for simultaneous moves and exact fast simulation.
- Athena (Sarantinos, arXiv:2212.13338; pkmn.ai summary): gen7 randbats agent using opponent modeling, rank #33. Direct evidence that opponent modeling can pay in randbats.
- PokéAgent Challenge and Metamon: learning approaches are strong but search dominates gen9.
- Smogon randbats guides: elite play emphasizes win-condition preservation, scouting, hidden information, switch prediction, not just raw damage.

Implications:

- A standalone eval must encode win conditions, endgames, PP, speed control, and hidden information.
- Exact generator belief is a unique randbats advantage.
- Ladder GXE, not H2H alone, is the final metric.

## Ranked eval design principles

1. Evaluate win probability / log-odds, not arbitrary uncalibrated points.
2. Evaluate belief summaries, not only concrete sampled worlds.
3. Count win conditions and remaining answers.
4. Add expected threat from unrevealed coverage.
5. Value speed control and priority correctly.
6. Value PP and recovery resources.
7. Make hazards conditional on actual roster, Boots, Magic Guard, removal, and forced switches.
8. Treat tera as option value, not just spent/not spent.
9. Add scouting value and concealment value.
10. Keep eval cheap, monotonic, debuggable, and tunable.

## Ranked tuning methods

1. Texel/logistic offline fitting on replay/self-play positions.
2. Paired H2H with Wilson CIs or SPRT.
3. SPSA for search-coupled weights.
4. Residual-error mining for missing features.
5. CLOP for low-dimensional knobs.
6. CMA-ES for small nonlinear blocks.
7. TDLeaf-style tuning only after simpler methods.
