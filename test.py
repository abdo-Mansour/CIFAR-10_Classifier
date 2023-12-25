C_Order = 10
num_of_features = int((C_Order + 1) * ( C_Order + 2) / 2)

# Extracting the features from the images
x_train_features = np.zeros(shape = (len(x_train), num_of_features))
x_test_features = np.zeros(shape = (len(x_test), num_of_features))

x_train_features = np.array([extract_image_features(x_train[i], C_Order) for i in range(len(x_train))])
x_test_features = np.array([extract_image_features(x_test[i], C_Order) for i in range(len(x_test))])


print(x_train_features.shape)
print(x_test_features.shape)
print(y_train.shape)
print(y_test.shape)


# Standardizing the data
scaler = StandardScaler()
x_train_features = scaler.fit_transform(x_train_features)
x_test_features = scaler.transform(x_test_features)



k_neighbours = 11

knn_model = KNeighborsClassifier(k_neighbours)

param_grid = {'n_neighbors': np.arange(1, 100)}

knn_grid = GridSearchCV(knn_model, param_grid)

knn_grid.fit(x_train_features, y_train)

print(f'Best score: {knn_grid.best_score_}')
print(f'Best parameters: {knn_grid.best_params_}')


k_neighbours = knn_grid.best_params_['n_neighbors']

knn_model = KNeighborsClassifier(k_neighbours)

knn_model.fit(x_train_features, y_train)

print(f'Test accuracy: {r2_score(x_test_features, y_test)}')



