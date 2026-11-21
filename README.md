# Bereinigte Entgeltlücke mit Machine Learning

![CI](https://github.com/mweyrauch/Entgeltanalyse/actions/workflows/ci.yml/badge.svg)

Die EU-Entgelttransparenzrichtlinie (EU) 2023/970 verlangt von Arbeitgebern, Entgeltunterschiede
zwischen Frauen und Männern zu berichten. Bleibt in einer Gruppe gleicher oder gleichwertiger Arbeit
ein Unterschied von 5 % oder mehr, der sich nicht durch objektive, geschlechtsneutrale Kriterien
erklären lässt, ist eine gemeinsame Entgeltbewertung fällig. Die Kernfrage ist also: **Wie groß ist
die Entgeltlücke bei gleichen objektiven Merkmalen?**

Dieses Projekt vergleicht, wie gut klassische und ML-gestützte Methoden diese bereinigte Lücke
treffen. Weil echte Personaldaten sensibel sind und die wahre Lücke nie bekannt ist, arbeitet es mit
**synthetischen Daten mit eingebauter, bekannter Lücke**. So lässt sich messen, wie weit jede
Methode danebenliegt.

## Kernergebnis

Bei nichtlinearer Entgeltstruktur überschätzt die übliche lineare Regression die Lücke um knapp
einen Prozentpunkt, und ihr 95-%-Intervall enthält den wahren Wert nur in 80 % der Fälle.
**Double Machine Learning trifft die Lücke in allen Szenarien bis auf 0,2 Prozentpunkte** und hält
die nominelle Intervallabdeckung ein. Keine Methode erkennt aber Benachteiligung, die über die
herausgerechneten Merkmale selbst wirkt (hier: seltenere Beförderung).

![Simulationsstudie](docs/figures/study.png)

| Szenario | Methode | Schätzung | Verzerrung | Abdeckung 95-%-Intervall |
|---|---|---:|---:|---:|
| linear | Lineare Regression | −4,8 % | +0,1 pp | 100 % |
| | ML-Oaxaca-Blinder | −5,3 % | −0,4 pp | 63 % |
| | Double Machine Learning | −4,8 % | +0,1 pp | 100 % |
| nichtlinear | Lineare Regression | −5,7 % | −0,8 pp | 80 % |
| | ML-Oaxaca-Blinder | −5,1 % | −0,3 pp | 60 % |
| | Double Machine Learning | −4,7 % | +0,2 pp | 100 % |
| + seltenere Beförderung | Lineare Regression | −6,0 % | −1,1 pp | 57 % |
| | ML-Oaxaca-Blinder | −4,8 % | 0,0 pp | 40 % |
| | Double Machine Learning | −4,7 % | +0,2 pp | 93 % |

Wahre direkte Lücke −4,9 %; im dritten Szenario beträgt die Gesamtlücke einschließlich seltenerer
Beförderung −9,6 %. Unbereinigt liegen die Werte bei −7,6 %, −8,5 % und −13,0 %. 3000 Beschäftigte
je simuliertem Unternehmen, **30 Wiederholungen** je Szenario (vorläufig; die Abdeckungswerte sind
daher auf etwa ±8 Prozentpunkte genau).

## Vorgehen

1. **Synthetische Daten** (`paygap/generate.py`): Jobfamilie, Standort, Alter, Berufserfahrung,
   Teilzeit, Entgeltstufe und Geschlecht; Zielgröße ist das logarithmierte Stundenentgelt. Frauen
   arbeiten häufiger in Teilzeit, in anderen Jobfamilien und haben im Mittel mehr
   Erwerbsunterbrechungen. Diese Unterschiede sind erklärt. Zusätzlich verdienen Frauen bei gleichen
   Merkmalen 5 Log-Punkte (≈ 4,9 %) weniger; das ist die direkte Lücke. Drei Szenarien:
   - **linear:** Entgelt hängt linear und additiv von den Merkmalen ab.
   - **nichtlinear:** Erfahrung wirkt sättigend, Stufe 4 bringt einen Sprung, der Standortzuschlag
     hängt von der Jobfamilie ab, der Teilzeitabschlag wächst mit der Entgeltstufe.
   - **seltenere Beförderung:** wie nichtlinear, zusätzlich erreichen Frauen bei gleicher
     Qualifikation seltener höhere Entgeltstufen.
2. **Schätzer** (`paygap/methods.py`):
   - **Unbereinigt:** Differenz der Mittelwerte.
   - **Lineare Regression:** Stundenentgelt auf Geschlecht und alle Merkmale, robuste
     Standardfehler (HC1). Der übliche Ansatz in Entgeltanalysen.
   - **ML-Oaxaca-Blinder:** Gradient Boosting lernt die Entgeltstruktur der Männer und sagt für
     jede Frau vorher, was ein Mann mit ihren Merkmalen verdienen würde. Die mittlere Abweichung
     ist der unerklärte Teil.
   - **Double Machine Learning:** Zwei Gradient-Boosting-Modelle sagen Entgelt und Geschlecht aus
     den Merkmalen vorher (mit 5-fachem Cross-Fitting). Die Lücke ergibt sich aus der Regression der
     Restgrößen aufeinander und hat ein gültiges Konfidenzintervall, obwohl die Modelle flexibel sind
     (Chernozhukov et al., 2018).
3. **Simulationsstudie** (`paygap/study.py`): je Szenario viele unabhängige Datensätze; gemessen
   werden Verzerrung, RMSE und wie oft das 95-%-Intervall die wahre Lücke enthält.
4. **Überlappungsdiagnose:** Das Geschlechtsmodell aus Double Machine Learning zeigt, wo es kaum
   vergleichbare Personen des anderen Geschlechts gibt (vorhergesagter Frauenanteil unter 5 % oder
   über 95 %). In den simulierten Daten betrifft das im Mittel 10 % der Beschäftigten, mit
   seltenerer Beförderung 15 %. Es sind vor allem Teilzeitkräfte in Forschung, Verwaltung und
   Vertrieb (fast nur Frauen) sowie Vollzeitkräfte in der Produktion (fast nur Männer).

## Einordnung

- **Warum ML hier hilft:** Die lineare Regression unterstellt eine Entgeltstruktur. Stimmt sie
  nicht und sind Frauen und Männer ungleich über die Merkmale verteilt, landet der Modellfehler in
  der Lücke. Flexible Modelle vermeiden das.
- **Warum nicht einfach ML-Oaxaca-Blinder:** Das Gradient-Boosting-Modell glättet in Bereichen mit
  wenigen Männern (Regularisierung) und ist dadurch leicht verzerrt. Sein Intervall berücksichtigt
  zudem nur das Rauschen der Frauen, nicht die Unsicherheit des Modells, und ist daher zu schmal.
  Double Machine Learning korrigiert beides.
- **Was keine Methode löst:** Wer die Entgeltstufe herausrechnet, obwohl Frauen seltener befördert
  werden, rechnet einen Teil der Benachteiligung weg. Welche Merkmale als objektiv gelten, ist eine
  inhaltliche Entscheidung, keine statistische. Eine bereinigte Lücke ist außerdem kein Beweis für
  Diskriminierung und keine Entlastung, sondern ein Hinweis, wo genauer hingesehen werden sollte.

## Grenzen

- Synthetische Daten: Die Ergebnisse zeigen, wie die Methoden unter den gewählten Annahmen
  arbeiten, nicht wie groß die Lücke in einem echten Unternehmen ist.
- Bisher wird nur eine Gesamtlücke geschätzt; die Richtlinie verlangt Auswertungen je Gruppe
  gleichwertiger Arbeit.
- Vorläufig 30 Wiederholungen je Szenario.

## Ausführen

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/analyze.py --scenario nonlinear --seed 1     # ein Datensatz, alle Schätzer
python scripts/run_study.py                                 # Studie, 100 Wiederholungen (~15 min)
python scripts/run_study.py --reps 5 --out-dir results/test # schneller Test
pytest -q
```

`analyze.py --csv datei.csv` analysiert eigene Daten mit den Spalten `female`, `job`, `location`,
`age`, `experience`, `parttime`, `level` und `log_wage`.

## Struktur

```
configs/default.yaml     Studienparameter und Modellparameter
src/paygap/generate.py   synthetische Personaldaten mit bekannter Lücke
src/paygap/methods.py    Schätzer: unbereinigt, OLS, ML-Oaxaca-Blinder, Double ML
src/paygap/study.py      Simulationsstudie und Kennzahlen
src/paygap/plots.py      Abbildung
scripts/                 analyze.py (ein Datensatz), run_study.py (Studie)
tests/                   pytest
docs/                    Abbildung und Kennzahlen des letzten Studienlaufs
```

## Nächste Schritte

- Lücke je Gruppe gleichwertiger Arbeit mit Bootstrap-Intervallen und der 5-%-Schwelle.
- Datenschutz: Mindestgruppengrößen oder Differential Privacy für kleine Gruppen.
- Auffällige Einzelfälle per Quantilregression.
