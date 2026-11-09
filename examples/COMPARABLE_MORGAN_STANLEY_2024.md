# Cas de marché identifiable — Morgan Stanley (émission réelle, 26 juin 2024)

**Source primaire** : [SEC, Morgan Stanley Finance LLC, Pricing Supplement No. 2,759, déposé le 26 juin 2024](https://www.sec.gov/Archives/edgar/data/895421/000183988224020018/ms2759_424b2-12083.htm).

Ce document est un **point de comparaison de marché vérifiable**, PAS une calibration du modèle ACRO.
Les hypothèses fictives 23/26/28 % et les corrélations 0,50/0,35/0,45 du projet
ne sont **pas** des volatilités/corrélations implicites de ce trio d'actions, ni une reconstitution de leur cotation de 2024.

| Élément | Produit réellement émis | Projet ACRO (configuration de référence) |
|---|---|---|
| Émetteur | Morgan Stanley Finance LLC (garanti par Morgan Stanley) | Émetteur théorique ; financement/crédit non modélisés |
| Sous-jacents | Airbnb (ABNB), Amazon (AMZN), Nvidia (NVDA) | Trois actions synthétiques **sans ticker** |
| Coupon | **18 % annualisé**, conditionnel, mensuel | **≈18 % au pair sous Q synthétique**, semestriel |
| Barrière de coupon | 60 % | 70 % |
| Rappel anticipé | Mensuel après période initiale de 3 mois, seuil 100 % | Annuel années 1–4, seuil 100 % |
| Maturité | 1er juillet 2027 (~3 ans après émission) | 5 ans |
| Barrière de capital finale | 60 % ; perte proportionnelle au pire sous-jacent si franchie | 60 % européenne ; perte proportionnelle au pire sous-jacent |
| Prix d'émission / valeur estimée émetteur | **1 000 $ / 948,50 $** par titre | 100 € au pair théorique (sans frais/marge émetteur) |
| Commission agent | 45 $ par titre de 1 000 $ | Non modélisée |
| Option ACRO | **Aucune** | Option distincte de renouvellement, prime payée en supplément |

## Ce que cela prouve — et ne prouve pas

- Un coupon conditionnel de l'ordre de 18 % **a été commercialisé** sur un worst-of de trois actions.
- **Il ne valide pas** le 18,2 % du Phoenix ACRO : dates, barrières, maturité, sous-jacents, coût de financement et régime de volatilité diffèrent.
- La différence entre prix payé (1 000 $) et valeur estimée (948,50 $) souligne l'importance des frais, coûts de distribution/hedging et conventions de financement ; ce n'est **pas** une marge de 51,50 $ qu'il faudrait automatiquement soustraire au modèle ACRO.
- L'émetteur réel est exposé à la volatilité implicite, au skew, aux dividendes, à la corrélation, à la solvabilité, aux coûts de couverture. Sans les données historiques des surfaces implicites au **26 juin 2024**, une reproduction quantitative indépendante de l'émission serait artificielle.

## Prochaine étape de calibration, si poursuivie

1. Obtenir les courbes taux/dividendes forward et surfaces de volatilité implicites ABNB/AMZN/NVDA **au 26 juin 2024**.
2. Choisir/calibrer une surface de dépendance multiactifs et un processus de volatilité sous Q ; documenter la calibration et l'erreur de réplication des options vanilles.
3. Implémenter fidèlement le calendrier **mensuel** du titre réel, et les coûts/valeur d'émission ; **ne pas** prétendre reproduire la note avec les seuls paramètres synthétiques actuels.
4. Séparer cette calibration du produit ACRO : le nouveau Phoenix et son option supplémentaire sont une autre structure.

**Formulation entretien :** « Pour tester la plausibilité des ordres de grandeur, je compare mon modèle synthétique à une vraie émission worst-of de Morgan Stanley à 18 % conditionnel. Les contrats ne sont pas identiques, et je ne prétends pas avoir calibré mes paramètres sur ce marché. »
