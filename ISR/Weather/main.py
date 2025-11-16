import requests
# sign up and get api key
#https://openweathermap.org/api                  

def get_weather(city_name, api_key):
    """Fetch and display live weather data for a given city."""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"  # Celsius
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["cod"] != 200:
            print(f"❌ City '{city_name}' not found. Please try again.")
            return

        # Extract data
        main = data["main"]
        wind = data["wind"]
        weather = data["weather"][0]

        print("\n=== 🌦 Live Weather Report ===")
        print(f"City: {data['name']}, {data['sys']['country']}")
        print(f"Temperature: {main['temp']}°C")
        print(f"Feels Like: {main['feels_like']}°C")
        print(f"Humidity: {main['humidity']}%")
        print(f"Wind Speed: {wind['speed']} m/s")
        print(f"Weather: {weather['main']}")
        print(f"Description: {weather['description'].capitalize()}")

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network Error: {e}")
    except KeyError:
        print("⚠️ Error reading weather data. Try another city.")


if __name__ == "__main__":
    print("=== Live Weather Info App ===")
    city = input("Enter city name: ").strip()
    api_key = "YOUR_API_KEY_HERE"  # Replace with your actual OpenWeatherMap key

    get_weather(city, api_key)
