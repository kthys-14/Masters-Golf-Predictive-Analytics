import requests
import json
import pandas as pd
from typing import List, Dict, Union, Optional, Any
import traceback

try:
    from data_golf import DataGolfClient
    DATAGOLF_PACKAGE_AVAILABLE = True
except ImportError:
    print("Warning: data-golf package not found. Using a minimal implementation.")
    DATAGOLF_PACKAGE_AVAILABLE = False
    # Provide a minimal implementation if the package is not available
    class DataGolfClient:
        def __init__(self, api_key, verbose=False):
            self.api_key = api_key
            self.verbose = verbose
            self.base_url = "https://feeds.datagolf.com"

class HistoricalData:
    """
    Class to handle historical data API endpoints
    """
    def __init__(self, client):
        self.client = client
        
        # The official data-golf package might store api_key differently
        if DATAGOLF_PACKAGE_AVAILABLE:
            # For official data-golf package
            # The API key is likely stored in the client's internal config
            if hasattr(client, '_config') and hasattr(client._config, 'api_key'):
                self.api_key = client._config.api_key
            # Try to extract from client.general._client.api_key
            elif hasattr(client, 'general') and hasattr(client.general, '_client') and hasattr(client.general._client, 'api_key'):
                self.api_key = client.general._client.api_key
            else:
                # Fall back to the API key provided in setup.py or directly in code
                self.api_key = 'baf700f24ceb315b87e9f65d98e9'
        else:
            # For our minimal implementation
            self.api_key = client.api_key
        
        # Get base URL and verbose setting
        self.base_url = getattr(client, 'base_url', "https://feeds.datagolf.com")
        self.verbose = getattr(client, 'verbose', False)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """
        Make a request to the DataGolf API
        
        Args:
            endpoint (str): API endpoint path
            params (Dict[str, Any]): Query parameters
            
        Returns:
            Dict: JSON response from the API
        """
        # Ensure API key is included
        if 'key' not in params:
            params['key'] = self.api_key
        
        # Ensure file format is specified
        if 'file_format' not in params:
            params['file_format'] = 'json'
        
        # Build the full URL
        url = f"{self.base_url}/{endpoint}"
        
        # Make the request
        if self.verbose:
            print(f"Making request to {url} with params {params}")
        
        response = requests.get(url, params=params)
        
        # Check for successful response
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # Parse JSON response
        if params.get('file_format') == 'json':
            try:
                return response.json()
            except json.JSONDecodeError:
                print(f"Error decoding JSON response: {response.text[:200]}...")
                return {"error": "Failed to decode JSON", "text": response.text[:200]}
        else:
            return response.text
    
    def historical_rounds(self, 
                        tour: str = 'pga',
                        event_id: str = 'all',
                        year: Union[str, int] = None,
                        file_format: str = 'json') -> Union[Dict, pd.DataFrame]:
        """
        Fetch historical raw round data from DataGolf API
        
        Args:
            tour (str): Tour code ('pga', 'euro', 'kft', etc.)
            event_id (str): Event ID or 'all' for all events
            year (Union[str, int]): Calendar year (2017-2025)
            file_format (str): Response format ('json' or 'csv')
            
        Returns:
            Union[Dict, pd.DataFrame]: JSON response or DataFrame with historical round data
        """
        if year is None:
            raise ValueError("Year must be specified")
        
        # Use the exact URL format from the instructions
        # https://feeds.datagolf.com/historical-raw-data/rounds?tour=[tour]&event_id=[event_id]&year=[year]&file_format=[file_format]&key=[key]
        params = {
            'tour': tour,
            'event_id': event_id,
            'year': str(year),
            'file_format': file_format,
            'key': self.api_key  # Include the key in params
        }
        
        # Direct URL construction as specified in the instructions
        url = f"{self.base_url}/historical-raw-data/rounds"
        
        if self.verbose:
            print(f"Making request to {url} with params: {params}")
        
        # Make the direct request
        response = requests.get(url, params=params)
        
        # Check for successful response
        if response.status_code != 200:
            print(f"API request failed with status code {response.status_code}")
            print(f"Response: {response.text[:500]}")
            raise Exception(f"API request failed with status code {response.status_code}")
        
        # Process based on format
        if file_format == 'json':
            try:
                result = response.json()
                return result
            except json.JSONDecodeError:
                print(f"Error decoding JSON response: {response.text[:200]}...")
                # Try to extract meaningful data from the response
                return {"error": "Failed to decode JSON", "text": response.text[:1000]}
        else:
            return response.text
    
    def _flatten_historical_data(self, data: Dict) -> pd.DataFrame:
        """
        Flatten the historical data structure into a DataFrame
        
        Args:
            data (Dict): Raw historical data from the API
            
        Returns:
            pd.DataFrame: Flattened DataFrame with all rounds
        """
        all_rounds = []
        
        # Debug the data structure
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            print(f"Keys: {list(data.keys())[:10]} {'...' if len(data.keys()) > 10 else ''}")
            
            # Extract a sample key
            sample_key = list(data.keys())[0]
            print(f"Sample key: {sample_key}, Type of value: {type(data[sample_key])}")
            
            # Check if all keys are event IDs (integers or numbers as strings)
            event_ids = [k for k in data.keys() if k.isdigit() or (isinstance(k, int))]
            print(f"Number of likely event IDs: {len(event_ids)}")
            
            # Process only keys that are event IDs
            for event_id in event_ids:
                event_data = data[event_id]
                
                if not isinstance(event_data, dict):
                    print(f"Skipping non-dictionary event data for event ID {event_id}")
                    continue
                    
                print(f"Processing event ID {event_id}, keys: {list(event_data.keys())}")
                
                # Extract event details
                event_name = event_data.get('event_name', f"Event {event_id}")
                event_completed = event_data.get('event_completed', 'Unknown')
                
                # Check if scores are available
                if 'scores' not in event_data or not isinstance(event_data['scores'], list):
                    print(f"No scores data for event ID {event_id}")
                    continue
                
                # Process each player's scores
                for player_data in event_data['scores']:
                    if not isinstance(player_data, dict):
                        continue
                        
                    player_name = player_data.get('player_name', 'Unknown Player')
                    dg_id = player_data.get('dg_id', None)
                    finish_position = player_data.get('fin_text', 'Unknown')
                    
                    # Process each round
                    for round_key in ['round_1', 'round_2', 'round_3', 'round_4']:
                        if round_key in player_data and isinstance(player_data[round_key], dict):
                            round_data = player_data[round_key]
                            
                            # Extract round details
                            round_num = int(round_key.split('_')[1])
                            
                            # Create a record for this round
                            round_record = {
                                'event_id': event_id,
                                'event_name': event_name,
                                'event_completed': event_completed,
                                'player_name': player_name,
                                'player_id': dg_id,
                                'finish_position': finish_position,
                                'round_num': round_num
                            }
                            
                            # Add all round statistics
                            round_record.update(round_data)
                            
                            all_rounds.append(round_record)
            
            print(f"Total rounds collected: {len(all_rounds)}")
        else:
            print(f"Unexpected data type: {type(data)}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        if not all_rounds:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_rounds)
        
        # Ensure consistent column names for key stats
        rename_cols = {
            'score': 'round_score',
            'sg_putt': 'sg_putt',
            'sg_arg': 'sg_arg',
            'sg_app': 'sg_app',
            'sg_ott': 'sg_ott',
            'sg_t2g': 'sg_t2g',
            'sg_total': 'sg_total'
        }
        
        # Only rename columns that exist
        rename_dict = {k: v for k, v in rename_cols.items() if k in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
            
        return df
    
    def fetch_all_pga_data(self, 
                           start_year: int = 2020, 
                           end_year: int = 2024,
                           event_id: str = 'all',
                           file_format: str = 'json') -> pd.DataFrame:
        """
        Fetch all PGA tour performances between specified years
        
        Args:
            start_year (int): Starting year (inclusive)
            end_year (int): Ending year (inclusive)
            event_id (str): Event ID or 'all' for all events
            file_format (str): Response format ('json' or 'csv')
            
        Returns:
            pd.DataFrame: Combined DataFrame with all historical round data
        """
        all_data = []
        
        for year in range(start_year, end_year + 1):
            try:
                print(f"Fetching data for year {year}...")
                year_data = self.historical_rounds(
                    tour='pga',
                    event_id=event_id,
                    year=year,
                    file_format=file_format
                )
                
                # Process the data based on format
                if file_format == 'json':
                    # Check the structure of the JSON
                    if isinstance(year_data, dict):
                        # Flatten the data structure into a DataFrame
                        df = self._flatten_historical_data(year_data)
                        if not df.empty:
                            df['year'] = year  # Add year column for reference
                            all_data.append(df)
                            print(f"Successfully processed {len(df)} rounds for {year}")
                        else:
                            print(f"Warning: No rounds found for year {year}")
                    else:
                        print(f"Warning: Invalid data format for year {year}")
                        
                else:  # CSV format
                    try:
                        df = pd.read_csv(pd.StringIO(year_data))
                        if not df.empty:
                            df['year'] = year
                            all_data.append(df)
                            print(f"Successfully processed {len(df)} rounds for {year}")
                        else:
                            print(f"Warning: No data found for year {year}")
                    except Exception as e:
                        print(f"Error processing CSV data for year {year}: {e}")
            
            except Exception as e:
                print(f"Error fetching data for year {year}: {e}")
                traceback.print_exc()
        
        # Combine all years into a single DataFrame
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Print stats about the data
            print(f"\nTotal rounds collected: {len(combined_df)}")
            print(f"Years: {combined_df['year'].nunique()} ({combined_df['year'].min()}-{combined_df['year'].max()})")
            print(f"Events: {combined_df['event_name'].nunique()}")
            print(f"Players: {combined_df['player_name'].nunique()}")
            
            return combined_df
        else:
            # Return empty DataFrame with expected columns
            print("Warning: No data was collected for any year")
            return pd.DataFrame(columns=['event_id', 'event_name', 'player_name', 'round_num', 
                                         'round_score', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'year'])

# Extend DataGolfClient with historical data functionality
def extend_datagolf_client(client, api_key=None):
    """
    Extend the DataGolfClient with historical data functionality
    
    Args:
        client (DataGolfClient): Existing DataGolfClient instance
        api_key (str, optional): API key to use if not available from client
        
    Returns:
        DataGolfClient: Extended client with historical data functionality
    """
    # Create the historical data handler
    historical_handler = HistoricalData(client)
    
    # If API key was explicitly provided, use it
    if api_key:
        historical_handler.api_key = api_key
        
    # Attach to the client
    client.historical = historical_handler
    return client

# Example usage
if __name__ == "__main__":
    # API key for DataGolf
    API_KEY = 'baf700f24ceb315b87e9f65d98e9'
    
    try:
        # Create a client
        client = DataGolfClient(api_key=API_KEY, verbose=True)
        
        # Extend the client with our historical data functionality
        client = extend_datagolf_client(client, api_key=API_KEY)
        
        # Test the functionality
        try:
            # Get a sample event ID first
            print("Fetching a sample from 2023 to examine structure...")
            data_sample = client.historical.historical_rounds(tour='pga', year=2023, event_id='all')
            
            if isinstance(data_sample, dict):
                print(f"Data structure: Dictionary with {len(data_sample)} keys")
                key_sample = list(data_sample.keys())[:5]
                print(f"Sample keys: {key_sample}")
                
                # Look at structure of a sample event
                if key_sample and key_sample[0].isdigit():
                    sample_event_id = key_sample[0]
                    sample_event = data_sample[sample_event_id]
                    
                    if isinstance(sample_event, dict):
                        print(f"\nSample event (ID: {sample_event_id}):")
                        for k, v in sample_event.items():
                            if k != 'scores':
                                print(f"  {k}: {v}")
                        
                        if 'scores' in sample_event and isinstance(sample_event['scores'], list):
                            print(f"  scores: List with {len(sample_event['scores'])} players")
                            
                            # Look at first player's data
                            if sample_event['scores']:
                                player = sample_event['scores'][0]
                                print(f"\nSample player data:")
                                for k, v in player.items():
                                    if not k.startswith('round_'):
                                        print(f"  {k}: {v}")
                                
                                # Look at first round data
                                if 'round_1' in player and isinstance(player['round_1'], dict):
                                    print(f"\nSample round data (round 1):")
                                    for k, v in player['round_1'].items():
                                        print(f"  {k}: {v}")
            
            # Try to process it and see if it works
            print("\nProcessing the sample data...")
            processed_df = client.historical._flatten_historical_data(data_sample)
            
            if not processed_df.empty:
                print(f"Successfully processed {len(processed_df)} rounds")
                print(f"Columns: {processed_df.columns.tolist()}")
                print("\nSample of processed data:")
                print(processed_df.head(3).to_string())
                
                # Save to CSV
                sample_file = 'pga_data_sample_2023.csv'
                processed_df.to_csv(sample_file, index=False)
                print(f"Saved sample data to {sample_file}")
                
                # Process 2020-2024 data (commented out for now)
                process_all = input("\nDo you want to process all data from 2020-2024? (y/n): ")
                if process_all.lower() == 'y':
                    print("\nFetching all PGA data from 2020-2024...")
                    all_data = client.historical.fetch_all_pga_data(
                        start_year=2020, 
                        end_year=2024
                    )
                    
                    if not all_data.empty:
                        print(f"Successfully collected {len(all_data)} rounds")
                        all_data_file = 'pga_data_2020_2024.csv'
                        all_data.to_csv(all_data_file, index=False)
                        print(f"Saved all data to {all_data_file}")
                else:
                    print("Skipping full data processing.")
            else:
                print("Failed to process the sample data.")
            
        except Exception as e:
            print(f"Error testing historical data functionality: {e}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error creating or extending the DataGolfClient: {e}")
        traceback.print_exc() 