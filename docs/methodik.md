# Methodik

## Was die wahre Lücke ist

Der Datengenerator berechnet das Stundenentgelt für alle Beschäftigten mit derselben Formel aus
Jobfamilie, Standort, Alter, Erfahrung, Teilzeit und Entgeltstufe. Bei Frauen wird danach ein fester
Abschlag von 0,05 Log-Punkten abgezogen (≈ −4,9 %). Das ist die **direkte Lücke**: Eine Frau verdient
4,9 % weniger als ein Mann mit genau denselben Merkmalen. In echten Daten ist dieser Wert unbekannt;
hier ist er bekannt, und deshalb lässt sich messen, wie weit jede Methode danebenliegt.

Im Szenario mit seltenerer Beförderung gibt es zusätzlich die **Gesamtlücke**: Für jede Frau wird
berechnet, was sie verdienen würde, wenn sie in allem wie ein Mann behandelt würde, also auch die
Entgeltstufe erreicht hätte, die ein Mann mit ihrer Qualifikation erreicht. Der Mittelwert dieser
Differenzen ist die Gesamtlücke (≈ −9,5 %). Weil die Methoden die Entgeltstufe als Merkmal
herausrechnen, können sie nur die direkte Lücke sehen.

## Grundidee von Double Machine Learning

**Modell.** Das logarithmierte Stundenentgelt y hängt so vom Geschlecht d (1 = Frau) und den
Merkmalen X ab:

    y = θ · d + g(X) + ε

g(X) ist die Entgeltstruktur und darf beliebig nichtlinear sein; θ ist die gesuchte Lücke. Die
lineare Regression unterstellt dagegen, dass auch g linear ist. Stimmt das nicht, landet der Fehler
in θ.

**Beide Seiten bereinigen.** Statt g direkt zu schätzen, lernen zwei ML-Modelle:

1. das erwartete Entgelt eines Profils, ℓ(X) = E[y | X], ohne das Geschlecht zu kennen.
   Restgröße u = y − ℓ̂(X): Wie viel verdient jemand mehr oder weniger als für das Profil üblich?
2. die Wahrscheinlichkeit, dass ein Profil eine Frau ist, m(X) = P(Frau | X).
   Restgröße v = d − m̂(X): Wie überraschend ist das Geschlecht für dieses Profil?

Die Lücke ist die Steigung der Regression von u auf v:

    θ̂ = Σ u·v / Σ v²

Das ist die ML-Variante des Frisch-Waugh-Lovell-Theorems aus der linearen Regression. Die Lücke wird
nur aus dem Teil der Variation geschätzt, den die Merkmale nicht erklären. Personen, deren Geschlecht
fast sicher aus den Merkmalen folgt (m̂ nahe 0 oder 1), tragen kaum bei, weil es für sie keine
vergleichbaren Personen des anderen Geschlechts gibt.

**Warum Modellfehler kaum schaden.** Der Fehler in θ̂ hängt ungefähr vom Produkt der Fehler beider
Modelle ab, nicht von jedem einzeln (Neyman-Orthogonalität). Liegen beide Modelle um 10 % daneben,
ist θ̂ nur um etwa 1 % verzerrt.

**Cross-Fitting.** Die Daten werden in 5 Teile geteilt. Die Vorhersagen für jeden Teil stammen aus
Modellen, die auf den anderen vier Teilen gelernt wurden. So lernt kein Modell das Rauschen der
Personen, deren Restgrößen es liefert.

**Konfidenzintervall.** Mit ψᵢ = (uᵢ − θ̂·vᵢ)·vᵢ ist der Standardfehler
SE = √( mittel(ψ²) / mittel(v²)² / n ), das Intervall θ̂ ± 1,96·SE.

## Warum nicht einfach ML-Oaxaca-Blinder

ML-Oaxaca-Blinder nutzt nur ein Modell: Es lernt die Entgeltstruktur der Männer und vergleicht jede
Frau mit der Vorhersage für ihr Profil. Fehler dieses Modells gehen eins zu eins in die Lücke ein.
Gradient Boosting glättet gerade in Bereichen mit wenigen Männern, und genau dort arbeiten viele
Frauen. Das Intervall berücksichtigt außerdem nur das Rauschen der Frauen, nicht die Unsicherheit
des Modells, und ist daher zu schmal.

## Voraussetzungen

- Alle relevanten objektiven Merkmale sind erfasst; fehlende Merkmale landen in θ.
- Die Lücke ist für alle gleich. Variiert sie, schätzt Double Machine Learning einen gewichteten
  Mittelwert, in dem Profile mit vielen vergleichbaren Personen beider Geschlechter stärker zählen.
- Es gibt vergleichbare Personen (Überlappung). m̂ wird bei 1 % und 99 % abgeschnitten, damit
  einzelne extreme Werte die Schätzung nicht dominieren.
- Die herausgerechneten Merkmale sind legitim. Was als Merkmal herausgerechnet wird, etwa die
  Entgeltstufe, kann keine Methode mehr als Lücke erkennen.

**Literatur:** Chernozhukov, V. et al. (2018): Double/debiased machine learning for treatment and
structural parameters. *The Econometrics Journal* 21(1), C1–C68.
