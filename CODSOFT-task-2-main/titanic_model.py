import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    titanic = pd.read_csv("Titanic-Dataset.csv")

    titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
    titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
    titanic['Sex'] = titanic['Sex'].map({'female': 1, 'male': 0})
    titanic = pd.get_dummies(titanic, columns=['Embarked'], drop_first=True)

    y = titanic['Survived']
    X = titanic.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = accuracy_score(y_test, preds)

    print("Model Performance Summary")
    print("==========================")
    for name, acc in results.items():
        print(f"{name:<20}: {acc:.4f}")

    best_model_name = max(results, key=results.get)
    print(f"\nTop Performing Model: {best_model_name}")

    best_model = models[best_model_name]
    final_preds = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, final_preds))

    cv_score = cross_val_score(best_model, X, y, cv=5).mean()
    print(f"Mean Cross-Validation Accuracy: {cv_score:.4f}")

    with open("model_performance.txt", "w") as f:
        f.write("Model Performance Summary\n")
        f.write("==========================\n")
        for name, acc in results.items():
            f.write(f"{name:<20}: {acc:.4f}\n")
        f.write(f"\nTop Performing Model: {best_model_name}\n")
        f.write(f"Mean Cross-Validation Accuracy: {cv_score:.4f}\n")

    gender_survival = titanic.groupby('Sex')['Survived'].value_counts().unstack()
    gender_survival.plot(kind='bar', color=['red', 'green'])
    plt.title("Gender-wise Survival Distribution")
    plt.xlabel("Sex (0 = Male, 1 = Female)")
    plt.ylabel("Passenger Count")
    plt.tight_layout()
    plt.savefig("gender_survival_chart.png")
    plt.show()

    pclass_avg_survival = titanic.groupby('Pclass')['Survived'].mean()
    pclass_avg_survival.plot(kind='bar', color='skyblue')
    plt.title("Average Survival Rate per Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Average Survival Rate")
    plt.tight_layout()
    plt.savefig("pclass_survival_rate.png")
    plt.show()

if __name__ == "__main__":
    main()
