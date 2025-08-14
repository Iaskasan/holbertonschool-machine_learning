## Calculus

# 📚 Mémo Dérivées — Bases & Conventions

## 1️⃣ Notations & Conventions

Ces écritures sont **universelles** et servent juste à savoir lire les formules.

- **\(f'(x)\)** → "f prime de x", dérivée de la fonction \(f\) par rapport à \(x\)
- **\(\frac{d}{dx} f(x)\)** → autre notation pour la dérivée de \(f(x)\)
- **\(\ln(x)\)** → logarithme naturel (base \(e \approx 2.718\))
- **\(\ln(a) - \ln(b) = \ln\left(\frac{a}{b}\right)\)** → propriété des logarithmes

---

## 2️⃣ Règles de base à retenir absolument

Ces formules reviennent tout le temps (comme les tables de multiplication) :

| Fonction \(f(x)\)        | Dérivée \(f'(x)\)        |
|--------------------------|--------------------------|
| \(x^n\)                  | \(n \cdot x^{n-1}\)      |
| \(\ln(x)\)               | \(\frac{1}{x}\)          |
| \(e^x\)                  | \(e^x\)                  |
| \(\sin(x)\)              | \(\cos(x)\)              |
| \(\cos(x)\)              | \(-\sin(x)\)             |

---

## 3️⃣ Règles de calcul de dérivées

Ces règles permettent de combiner les dérivées de base pour créer des calculs plus complexes.

### ➡ Règle du produit
\[
\frac{d}{dx}[u \cdot v] = u' \cdot v + u \cdot v'
\]

### ➡ Règle du quotient
\[
\frac{d}{dx}\left[\frac{u}{v}\right] = \frac{u'v - uv'}{v^2}
\]

### ➡ Règle de la chaîne
\[
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
\]

---

## 4️⃣ Exemple simple

**Calculer** :
\[
\frac{d}{dx} \big[ x \cdot \ln(x) \big]
\]

**Solution** :
1. On pose \(u = x\), \(v = \ln(x)\)
2. \(u' = 1\), \(v' = \frac{1}{x}\)
3. On applique la règle du produit :
\[
u'v + uv' = 1 \cdot \ln(x) + x \cdot \frac{1}{x}
\]
4. On simplifie :
\[
\ln(x) + 1
\]

**Résultat** :
\[
\frac{d}{dx}[x \ln(x)] = \ln(x) + 1
\]

---

## 5️⃣ À retenir

- **Notations** → juste une façon d’écrire (à reconnaître, pas à calculer)
- **Formules de base** → à connaître par cœur
- **Résultats complexes** → toujours obtenus en appliquant les règles
