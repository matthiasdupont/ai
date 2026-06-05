const https = require('https');
function getWeatherinParis() {
    const apiUrl = "https://api.open-meteo.com/v1/forecast?latitude=48.8566&longitude=2.3522&current=temperature_2m,weather_code,wind_speed_10m,wind_direction_10m,precipitation";

    https.get(apiUrl, (res) => {
        let data = '';
        res.on('data', (chunk) => {
            data += chunk;
        });

        res.on('end', () => {
            try {
                if (res.statusCode !== 200) {
                    console.error(`Error fetching weather data: HTTP ${res.statusCode} - ${data}`);
                    return;
                }

                const weatherData = JSON.parse(data);
                if (!weatherData || !weatherData.current) {
                    console.error('Error parsing weather data: missing current field in API response');
                    console.error('Received response structure:', weatherData);
                    return;
                }

                const current = weatherData.current;
                console.log(`Current temperature in Paris: ${current.temperature_2m ?? 'N/A'}°C`);
                console.log(`Current weather code: ${current.weather_code ?? 'N/A'}`);
                console.log(`Current wind speed: ${current.wind_speed_10m ?? 'N/A'} km/h`);
                console.log(`Current wind direction: ${current.wind_direction_10m ?? 'N/A'}°`);
                console.log(`Current precipitation: ${current.precipitation ?? 'N/A'} mm`);
            } catch (error) {
                console.error('Error parsing weather data:', error);
            }

        });
    }).on('error', (err) => {
        console.error('Error fetching weather data:', err);
    });
}
getWeatherinParis();

