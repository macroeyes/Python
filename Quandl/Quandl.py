"""
Quandl's API for Python.

Currently supports getting, searching, and pushing datasets.

"""
from __future__ import print_function, division, absolute_import
import pickle
import datetime
import json
import pandas as pd
import re

from pprint import pprint
from pandas import DataFrame, Series
from dateutil import parser
from numpy import genfromtxt

try:
    from urllib.error import HTTPError  # Python 3
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    strings = str
except ImportError:
    from urllib import urlencode  # Python 2
    from urllib2 import HTTPError, Request, urlopen
    strings = unicode


#Base API call URL
QUANDL_API_URL = 'http://www.quandl.com/api/v1/'


def get(dataset, authtoken=None, returns='pandas', text=True, trim_start=None, trim_end=None,
        collapse=None, transformation=None, rows=None, sort_order='asc', **kwargs):
    """
    Return dataframe of requested dataset from Quandl.

    N.B. Note that Pandas expects timeseries data to be sorted ascending for most timeseries functionality to work.
    N.B. Any other `kwargs` passed to `get` are sent as field/value params to Quandl with no interference.

    :param dataset: str|list, depending on single dataset usage or multiset usage. Dataset codes are available on the Quandl website
    :param authtoken: str, Quandl API authentication token, (alias `auth_token`)
    :param trim_start, trim_end: str, Optional datefilers, otherwise entire dataset is returned
    :param collapse: str, Options are daily, weekly, monthly, quarterly, annual
    :param transformation: str, options are diff, rdiff, cumul, and normalize
    :param rows: int, Number of rows which will be returned
    :param sort_order: str, options are asc, desc. Default: `asc`
    :param returns: str, specify what format you wish your dataset returned as, either 'numpy' for a numpy ndarray, 'pandas' for a pandas DataFrame, 'json', 'xml', 'csv', 'plain' for raw data. Default: 'pandas'
    :param text: bool, specify whether to print output text to stdout, pass text=False to supress output.

    :rtype DataFrame|numpy.ndarray|dict
    """

    auth_token = authtoken
    pickled_auth = _getauthtoken(auth_token, text=text)
    trim_start = _parse_dates(trim_start)
    trim_end = _parse_dates(trim_end)

    # Check whether dataset is given as a string (for a single dataset) or an array (for a multiset call)
    if isinstance(dataset, (strings, str)):
        if returns not in ['pandas', 'numpy']:
            extension = returns
        else:
            extension = 'csv'
        url = QUANDL_API_URL + 'datasets/{}.{}?'.format(dataset, extension)
    elif dataset is list:
        url = QUANDL_API_URL + 'multisets.csv?columns='
        # Format for multisets call
        dataset = [d.replace('/', '.') for d in dataset]
        for i in dataset:
            url += i + ','
        # remove trailing ,
        url = url[:-1] + '&'
    else:
        raise Exception("Your dataset must either be specified as a string (containing a "
                        "Quandl code) or an array (of Quandl codes) for multisets")

    # Append all parameters to API call
    url = _append_query_fields(
        url,
        auth_token=pickled_auth,
        trim_start=trim_start,
        trim_end=trim_end,
        collapse=collapse,
        transformation=transformation,
        sort_order=sort_order,
        rows=rows,
        **kwargs
    )
    
    # handle various `returns` types, starting with returns='numpy'
    if returns == 'numpy':
        try:
            url = urlopen(url)
            try:
                # noinspection PyTypeChecker
                return genfromtxt(url, names=True, delimiter=',', dtype=None)
            except ValueError as e:
                raise Exception("Currently we only support multisets with up to 100 columns. \n"
                                "Please contact connect@quandl.com if this is a problem.")
        except IOError as e:
            raise Exception("Parsing Error on url: {}\n{}".format(url, e))
        except HTTPError as html_error:
            raise html_error
    elif returns == 'pandas':
        try:
            urldata = _download(url)
            if urldata.columns.size > 100:
                raise Exception("Currently we only support multisets with up to 100 columns. \n"
                                "Please contact connect@quandl.com if this is a problem.")
            if text:
                print("Returning Dataframe {}".format(dataset))
            return urldata
        except HTTPError as html_error:
            raise html_error
    else:
        try:
            return _download(url, raw=True)
        except HTTPError as html_error:
            raise html_error


def push(data, code, name, authtoken, desc='', override=False, text=True):
    """
    Upload a pandas Dataframe to Quandl and returns link to the dataset.

    :param data: Pandas ts or numpy array
    :param code: str, Dataset code must consist of only capital letters, numbers, and underscores
    :param name: str, Dataset name
    :param authtoken: str, Required to upload data
    :param desc: str, Description of dataset (optional, default='')
    :param override: bool, whether to overide dataset of same code (optional, default=False)
    :param text: bool, specify whether to print output prints to stdout (optional, default=True)

    :rtype str: url of uploaded dataset
    """

    override = str(override).lower()
    pickled_token = _getauthtoken(authtoken, text=text)
    if not pickled_token:
        raise Exception("You need an API token to upload your data to Quandl. \n"
                        "Please see www.quandl.com/API for more information.")

    #test that code is acceptable format
    _pushcodetest(code)
    datestr = ''

    # Verify and format the data for upload.
    if not isinstance(data, DataFrame):
        raise ValueError("Only pandas DataFrames are accepted for upload at this time")

    # check if indexed by date
    data_interm = data.to_records()
    index = data_interm.dtype.names
    datestr += ','.join(index) + '\n'

    #format data for uploading
    for i in data_interm:
        # Check if index is a date
        if isinstance(i[0], datetime.datetime):
            datestr += i[0].date().isoformat()
        else:
            try:
                datestr += _parse_dates(str(i[0]))
            except ValueError:
                raise Exception("Please check your indices, one of them is not a recognizable date")

        for n in i:
            if isinstance(n, (float, int)):
                datestr += ',' + str(n)
        datestr += '\n'

    params = {
        'name': name,
        'code': code,
        'description': desc,
        'update_or_create': override,
        'data': datestr
    }

    url = ''.join([QUANDL_API_URL, 'datasets.json?auth_token=', pickled_token])

    json_response = _htmlpush(url, params)
    if json_response['errors'] and json_response['errors']['code'][0] == 'has already been taken':
        raise ValueError("You are trying to overwrite a dataset which already \n"
                         "exists on Quandl. If this is what you wish to do please \n"
                         "recall the function with overide=True")

    # return URL of uploaded dataset
    return '/'.join(['http://www.quandl.com', json_response['source_code'], json_response['code']])


def search(query, source=None, page=1 , authtoken=None, text=True, raw=False, **kwargs):
    """
    Return array of dictionaries of search results.

    :param query: str, query to search with (required)
    :param source: str, source to search (optional, default=None)
    :param page: int, page number of search (optional, default=1)
    :param authtoken: str, Quandl auth token for extended API access (optional, default=None)
    :param text: bool, pecify whether to print output text to stdout, pass text=False to supress
    output.
    :param raw: bool, Return (nearly) raw Quandl API response body (optional, default=False)
    :param kwargs: dict, dictionary to be url_encoded and passed as url params (optional)

    :rtype list: search results
    """

    token = _getauthtoken(authtoken, text=text)
    search_url = 'http://www.quandl.com/api/v1/datasets.json?query='

    # parse query for proper API submission
    parsedquery = re.sub(" ", "+", query)
    parsedquery = re.sub("&", "+", parsedquery)
    url = search_url + parsedquery

    # use authtoken if present
    if token:
        url += '&auth_token=' + token

    # add search source_code if given
    if source:
        url += '&source_code=' + source

    # pass any additional kwargs as url params (future-proofing)
    if kwargs:
        url += "&{}".format(urlencode(kwargs))

    # page of results to return
    url += '&page=' + str(page)

    # make request, parse response as json
    data = json.loads(urlopen(url).read().decode("utf-8"))

    if raw:
        # full response
        return_data = data
    else:
        # nothing like full response ...
        try:
            datasets = data['docs']
        except TypeError:
            # @TODO: is raising an exception here correct/helpful?
            # (it might be nicer if it just returned None, [], or en empty Dataframe)
            raise TypeError("There are no matches for this search")

        return_data = []
        for i in range(len(datasets)):
            cleaned_dataset = {
                'name': datasets[i]['name'],
                'code': "{}/{}".format(datasets[i]['source_code'], datasets[i]['code']),
                'description': datasets[i]['description'], 'freq': datasets[i]['frequency'],
                'column_names': datasets[i]['column_names']
            }
            return_data.append(cleaned_dataset)
            if text and i < 4:
                pprint(cleaned_dataset, indent=4)

    return return_data


def _parse_dates(date):
    """
    Format date, returns None if date=None
    :param date: datetime.datetime|datetime.date|None
    :raise ValueError:
    :rtype : None|str
    """
    if isinstance(date, datetime.datetime):
        return_date = date.date().isoformat()
    elif isinstance(date, datetime.date):
        return_date = date.isoformat()
    elif date is None:
        return_date = None
    else:
        try:
            date = parser.parse(date)
            return_date = date.date().isoformat()
        except ValueError:
            raise ValueError("{} is not recognised a date.".format(date))

    return return_date

def _download(url, raw=False):
    """
    Download data into pandas dataframe, or get raw response
    :param url: str
    :param raw: bool
    :rtype : str|DataFrame
    """
    if raw:
        request = Request(url)
        return urlopen(request).read()
    else:
        return pd.read_csv(url, index_col=0, parse_dates=True)

def _htmlpush(url, raw_params):
    """
    Push data to Quandl
    :param url: str
    :param raw_params: dict
    :rtype : dict
    """
    params = urlencode(raw_params)
    response = urlopen(Request(url, params))
    return json.loads(response.read())

def _pushcodetest(code):
    """
    Test if code is capitalized alphanumeric
    :param code: str
    :raise Exception:
    :rtype : str
    """
    regex = re.compile('[^0-9A-Z_]')
    if regex.search(code):
        raise Exception("Your Quandl Code for uploaded data must consist of only \n"
                        "capital letters, underscores and numbers.")
    return code

def _getauthtoken(token_string, text=False):
    """
    Return and save API token to a pickle file for reuse.
    :param token_string: str, Quandl API token
    :param text: bool, Print output to stdout, default=False
    :rtype : str
    """
    try:
        with open('authtoken.p', 'rb') as pickle_file:
            savedtoken = pickle.load(pickle_file)
    except IOError:
        savedtoken = False
    if token_string:
        try:
            with open('authtoken.p', 'wb') as auth_token:
                pickle.dump(token_string, auth_token)
            if text:
                print("Token {} activated and saved for later use.".format(token_string))
        except Exception as e:
            raise Exception("Error writing token to cache:{}".format(e))
    elif savedtoken and not token_string:
        token_string = savedtoken
        if text:
            print("Using cached token {} for authentication.".format(token_string))
    else:
        if text:
            print("No authentication tokens found: usage will be limited.\nSee www.quandl.com/api for more information.")

    return token_string

def _append_query_fields(url, **kwargs):
    """
    In lieu of urllib's urlencode, as this handles None values by ignoring them.
    :param url: str
    :param kwargs: dict
    :rtype : str
    """
    return url + '&'.join(['{}={}'.format(key, val) for key, val in kwargs.items() if val])
