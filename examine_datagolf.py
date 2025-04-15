try:
    from data_golf import DataGolfClient
    
    # Initialize client with API key
    client = DataGolfClient(api_key='baf700f24ceb315b87e9f65d98e9', verbose=True)
    
    # Print available modules and methods
    print("DataGolfClient attributes:")
    for attr in dir(client):
        if not attr.startswith('_'):
            print(f"  - {attr}")
            
    # Print attributes of submodules if they exist
    if hasattr(client, 'general'):
        print("\nGeneral module methods:")
        for attr in dir(client.general):
            if not attr.startswith('_'):
                print(f"  - {attr}")
                
    if hasattr(client, 'predictions'):
        print("\nPredictions module methods:")
        for attr in dir(client.predictions):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    
    # Example call to test functionality
    try:
        print("\nTesting player_list endpoint:")
        result = client.general.player_list()
        print(f"  Result: {type(result)}, containing {len(result) if result else 0} items")
    except Exception as e:
        print(f"  Error calling player_list: {e}")
        
except ImportError as e:
    print(f"Error importing data-golf: {e}")
    print("The data-golf package is not installed. Installing manually would require:")
    print("pip install data-golf")
    
    # Suggest implementing a custom client
    print("\nWe need to implement a custom DataGolfClient to handle historical raw data fetching.")
except Exception as e:
    print(f"Error: {e}") 