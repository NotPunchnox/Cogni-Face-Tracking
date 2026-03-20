### Seuil de hystérésis
Pour éviter les tremblements, on utilise un seuil de 50px:
- Distance < -50px → "GAUCHE"
- Distance entre -50 et +50px → "CENTRE" (zone morte)
- Distance > +50px → "DROITE"

C'est une hystérésis qui prévient l'oscillation autour du centre.

### Calcul de confiance (distance euclidienne)
```
dist² = Δx² + Δy²
confiance = max(0, min(1, 1 - dist/dist_max))
```

### Conversion pixel→degré
Pour une FoV de 60° sur 640 pixels:
```
60° / 640 px ≈ 0.0937°/px ≈ 0.1°/px
```

### Oscillation du filtre
L'équation du smoothing:
```
output_n = output_(n-1) + k × (target - output_(n-1))
où k = smoothing_factor
```
Cela crée un filtre IIR qui amortit l'oscillation.

---

**Créé le:** 20 mars 2026  
**Projet:** Cogni-Project - Brain
