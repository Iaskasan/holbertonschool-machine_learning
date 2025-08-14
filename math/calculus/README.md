# 📚 Mémo Dérivées — Bases & Conventions

## 1️⃣ Notations & Conventions

Ces écritures sont **universelles** et servent juste à savoir lire les formules.

- `f'(x)` → "f prime de x", dérivée de la fonction f par rapport à x
- `d/dx f(x)` → autre notation pour la dérivée de f(x)
- `ln(x)` → logarithme naturel (base e ≈ 2.718)
- `ln(a) - ln(b) = ln(a/b)` → propriété des logarithmes

---

## 2️⃣ Règles de base à retenir absolument

Ces formules reviennent tout le temps (comme les tables de multiplication) :

| Fonction f(x)    | Dérivée f'(x)       |
|------------------|--------------------|
| x^n              | n * x^(n - 1)      |
| ln(x)            | 1 / x              |
| e^x              | e^x                |
| sin(x)           | cos(x)              |
| cos(x)           | -sin(x)             |

---

## 3️⃣ Règles de calcul de dérivées

Ces règles permettent de combiner les dérivées de base pour créer des calculs plus complexes.

### ➡ Règle du produit
d/dx [u * v] = u' * v + u * v'

### ➡ Règle du quotient
d/dx [u / v] = (u' * v - u * v') / v^2

### ➡ Règle de la chaîne
d/dx f(g(x)) = f'(g(x)) * g'(x)

---

## 4️⃣ Exemple simple

**Calculer** :  
d/dx [x * ln(x)]

**Solution** :
1. On pose u = x, v = ln(x)
2. u' = 1, v' = 1/x
3. On applique la règle du produit :  
   u' * v + u * v' = 1 * ln(x) + x * (1/x)
4. On simplifie :  
   ln(x) + 1

**Résultat** :  
d/dx [x * ln(x)] = ln(x) + 1

---

## 5️⃣ À retenir

- **Notations** → juste une façon d’écrire (à reconnaître, pas à calculer)
- **Formules de base** → à connaître par cœur
- **Résultats complexes** → toujours obtenus en appliquant les règles
