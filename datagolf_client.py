"""
DataGolf API Client Extension
----------------------------
Extension to the DataGolf API client with historical data endpoints.
"""

import requests
import json
import pandas as pd
from typing import Dict, List, Union, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('datagolf_client')

try:
    from data_golf import DataGolfClient
    DATAGOLF_PACKAGE_AVAILABLE = True
    logger.info("Official data-golf package found and imported.")
except ImportError:
    logger.warning("Official data-golf package not found. Using minimal implementation.")
    DATAGOLF_PACKAGE_AVAILABLE = False
    
    # Provide a minimal implementation if the package is not available
    class DataGolfClient:
        """
        Minimal implementation of DataGolfClient
        """
        def __init__(self, api_key, verbose=False):
            self.api_key = api_key
            self.verbose = verbose
            self.base_url = "https://feeds.datagolf.com"


class HistoricalDataClient:
    """
    Client for historical DataGolf API endpoints
    """
    def __init__(self, client):
        """
        Initialize the historical data client
        
        Args:
            client: DataGolfClient instance
        """
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
                # Get it from the client directly or use default
                self.api_key = getattr(client, 'api_key', None)
                if not self.api_key:
                    raise ValueError("API key not found in client. Please provide one explicitly.")
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
            logger.info(f"Making request to {url} with params {params}")
        
        response = requests.get(url, params=params)
        
        # Check for successful response
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            raise Exception(f"API request failed with status code {response.status_code}: {response.text[:200]}")
        
        # Parse JSON response
        if params.get('file_format') == 'json':
            try:
                return response.json()
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON response: {response.text[:200]}...")
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
            logger.info(f"Making request to {url} with params: {params}")
        
        # Make the direct request
        response = requests.get(url, params=params)
        
        # Check for successful response
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            raise Exception(f"API request failed with status code {response.status_code}")
        
        # Process based on format
        if file_format == 'json':
            try:
                result = response.json()
                return result
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON response: {response.text[:200]}...")
                # Try to extract meaningful data from the response
                return {"error": "Failed to decode JSON", "text": response.text[:1000]}
        else:
            return response.text
    
    def process_historical_data(self, data: Dict) -> pd.DataFrame:
        """
        Process the historical data structure into a DataFrame
        
        Args:
            data (Dict): Raw historical data from the API
            
        Returns:
            pd.DataFrame: Flattened DataFrame with all rounds
        """
        all_rounds = []
        
        if not isinstance(data, dict):
            logger.error(f"Unexpected data type: {type(data)}")
            return pd.DataFrame()
        
        # Debug the data structure
        if self.verbose:
            logger.info(f"Data type: {type(data)}")
            logger.info(f"Number of keys: {len(data)}")
        
        # Check if all keys are event IDs (integers or numbers as strings)
        event_ids = [k for k in data.keys() if k.isdigit() or (isinstance(k, int))]
        
        if self.verbose:
            logger.info(f"Number of likely event IDs: {len(event_ids)}")
        
        # Process only keys that are event IDs
        for event_id in event_ids:
            event_data = data[event_id]
            
            if not isinstance(event_data, dict):
                if self.verbose:
                    logger.warning(f"Skipping non-dictionary event data for event ID {event_id}")
                continue
                
            if self.verbose:
                logger.info(f"Processing event ID {event_id}")
            
            # Extract event details
            event_name = event_data.get('event_name', f"Event {event_id}")
            event_completed = event_data.get('event_completed', 'Unknown')
            
            # Check if scores are available
            if 'scores' not in event_data or not isinstance(event_data['scores'], list):
                if self.verbose:
                    logger.warning(f"No scores data for event ID {event_id}")
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
    
    def fetch_multi_year_data(self, 
                           start_year: int = 2020, 
                           end_year: int = 2024,
                           tour: str = 'pga',
                           event_id: str = 'all',
                           file_format: str = 'json') -> pd.DataFrame:
        """
        Fetch multi-year tour performances 
        
        Args:
            start_year (int): Starting year (inclusive)
            end_year (int): Ending year (inclusive)
            tour (str): Tour code ('pga', 'euro', 'kft', etc.)
            event_id (str): Event ID or 'all' for all events
            file_format (str): Response format ('json' or 'csv')
            
        Returns:
            pd.DataFrame: Combined DataFrame with all historical round data
        """
        all_data = []
        
        for year in range(start_year, end_year + 1):
            try:
                logger.info(f"Fetching data for {tour} tour, year {year}...")
                year_data = self.historical_rounds(
                    tour=tour,
                    event_id=event_id,
                    year=year,
                    file_format=file_format
                )
                
                # Process the data based on format
                if file_format == 'json':
                    # Check the structure of the JSON
                    if isinstance(year_data, dict):
                        # Flatten the data structure into a DataFrame
                        df = self.process_historical_data(year_data)
                        if not df.empty:
                            df['year'] = year  # Add year column for reference
                            all_data.append(df)
                            logger.info(f"Successfully processed {len(df)} rounds for {year}")
                        else:
                            logger.warning(f"No rounds found for year {year}")
                    else:
                        logger.warning(f"Invalid data format for year {year}")
                        
                else:  # CSV format
                    try:
                        df = pd.read_csv(pd.StringIO(year_data))
                        if not df.empty:
                            df['year'] = year
                            all_data.append(df)
                            logger.info(f"Successfully processed {len(df)} rounds for {year}")
                        else:
                            logger.warning(f"No data found for year {year}")
                    except Exception as e:
                        logger.error(f"Error processing CSV data for year {year}: {e}")
            
            except Exception as e:
                logger.error(f"Error fetching data for year {year}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Combine all years into a single DataFrame
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Print stats about the data
            logger.info(f"Total rounds collected: {len(combined_df)}")
            if not combined_df.empty:
                logger.info(f"Years: {combined_df['year'].nunique()} " + 
                         f"({combined_df['year'].min()}-{combined_df['year'].max()})")
                logger.info(f"Events: {combined_df['event_name'].nunique()}")
                logger.info(f"Players: {combined_df['player_name'].nunique()}")
            
            return combined_df
        else:
            # Return empty DataFrame with expected columns
            logger.warning("No data was collected for any year")
            return pd.DataFrame(columns=['event_id', 'event_name', 'player_name', 'round_num', 
                                         'round_score', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'year'])


def extend_datagolf_client(client: Any, api_key: Optional[str] = None) -> Any:
    """
    Extend a DataGolfClient instance with historical data functionality
    
    Args:
        client: DataGolfClient instance to extend
        api_key (str, optional): API key to use if not available from client
        
    Returns:
        Extended DataGolfClient with historical data functionality
    """
    # Create the historical data handler
    historical_client = HistoricalDataClient(client)
    
    # If API key was explicitly provided, use it
    if api_key:
        historical_client.api_key = api_key
        
    # Attach to the client
    client.historical = historical_client
    return client


def create_client(api_key: str, verbose: bool = False) -> Any:
    """
    Create an extended DataGolfClient with historical data functionality
    
    Args:
        api_key (str): DataGolf API key
        verbose (bool): Whether to enable verbose logging
        
    Returns:
        Extended DataGolfClient instance
    """
    try:
        # Create the base client
        client = DataGolfClient(api_key=api_key, verbose=verbose)
        
        # Extend with historical functionality
        extended_client = extend_datagolf_client(client, api_key=api_key)
        
        return extended_client
    
    except Exception as e:
        logger.error(f"Error creating DataGolfClient: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    API_KEY = 'baf700f24ceb315b87e9f65d98e9'
    
    try:
        client = create_client(API_KEY, verbose=True)
        
        # Fetch a small sample of data (one event from 2023)
        print("\nSample API call: fetching 2023 Masters Tournament data...")
        
        # Masters is typically event_id 23
        masters_data = client.historical.historical_rounds(
            tour='pga', 
            year=2023, 
            event_id='23'
        )
        
        # Process the data
        if isinstance(masters_data, dict):
            # Check for Masters specifically
            if '23' in masters_data:
                print(f"Found 2023 Masters data!")
                print(f"Event name: {masters_data['23'].get('event_name', 'Unknown')}")
                print(f"Players: {len(masters_data['23'].get('scores', []))}")
                
                # Process into a DataFrame
                print("\nProcessing data...")
                masters_df = client.historical.process_historical_data(masters_data)
                print(f"Processed {len(masters_df)} rounds")
                
                if not masters_df.empty:
                    print("\nSample data fields:")
                    print(masters_df.columns.tolist())
                    
                    print("\nSample player round:")
                    print(masters_df.iloc[0].to_dict())
                    
                    # Save to CSV for review
                    output_file = 'masters_2023_data.csv'
                    masters_df.to_csv(output_file, index=False)
                    print(f"\nSaved data to {output_file}")
        
        print("\nTo fetch data for all PGA rounds from 2020-2024, use:")
        print("client.historical.fetch_multi_year_data(start_year=2020, end_year=2024)")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc() 