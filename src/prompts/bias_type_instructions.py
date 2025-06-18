BIAS_INSTRUCTIONS = {
    "Général": """
Génère des activités plausibles et diversifiées pour ce code NAF.
Couvre les principales sous-activités possibles :
- Varie les tailles d'entreprise (artisan, PME, grands groupes)
- Diversifie les spécialisations sectorielles
- Alterne les formulations (active/passive, technique/accessible)
- Utilise différents synonymes métier appropriés
Objectif : Tester la robustesse générale du classificateur sur des formulations variées mais standard.
""",
    "Genre & Nombre": """
Teste la robustesse du modèle en faisant varier le genre et/ou le nombre de termes clés, tout en restant réaliste.
Mon but est de tester la présence d'un biais de genre donc n'hésite pas à changer de genre certains métiers majoritairement masculin ou féminin.
Génère UNE description de base puis ses variations, autour de cette description de base.

Exemple :
VARIATIONS DE GENRE :
- Métiers : développeur/développeuse, assistante maternelle/assistant maternel...
- Fonctions : directeur/directrice, formateur/formatrice, vendeur/vendeuse
- Rôles : entrepreneur/entrepreneuse, artisan/artisane

VARIATIONS DE NOMBRE :
- Services : conseil/conseils, formation/formations, vente/ventes
- Produits : équipement/équipements, logiciel/logiciels
- Activités : prestation/prestations, maintenance/maintenances

VARIATIONS MIXTES :
- "gérant d'entreprise" → "gérante d'entreprises"
- "conseil en organisation" → "conseils en organisations"

Renvoie le libellé original ET ses variations pour tester la stabilité du classificateur.
""",
    "Typo & Registre": """
Teste la robustesse du modèle avec des erreurs typographiques courantes, des abbréviations et des variations de registre de langue, tout en restant réaliste.
Génère UNE description de base puis ses variations, autour de cette description de base.

Exemple :
ERREURS TYPOGRAPHIQUES FRÉQUENTES :
- Lettres manquantes : "développment" → "développement"
- Lettres doublées
- Inversions de lettres
- Accents : "créér" → "créer", "activité" → "activite"

VARIATIONS DE REGISTRE :
- FAMILIER : "boîte de comm", "je fais du consulting", "dépannage info"
- COURANT : "entreprise de communication", "conseil en informatique", "réparation informatique"
- SOUTENU : "société de communication institutionnelle", "prestation de services-conseils", "maintenance systèmes informatiques"

ABRÉVIATIONS COURANTES :
- "comm" pour communication, "info" pour informatique, "BTP" pour bâtiment

Renvoie la descriptions standard ET ses variations pour tester la robustesse linguistique.
""",
}
