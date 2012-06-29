data = gamrnd(5,1,[1 5000]);

figure()
histnorm(data,50)

x = linspace(min(data), max(data), 100);
g = Gamma(5,1, 0)
hold on
plot(x, gampdf(x, 5,10), 'r')
plot(x, g.pdf(x), 'g')
hold off

gamfit(data)


g.fit_weighted(data, ones(5000))
g