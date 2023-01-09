# Databricks notebook source
# MAGIC %md
# MAGIC 1. Login to https://console.cloud.google.com/ . Enable Gmail API's
# MAGIC 2. Create "Credentials". Create "Service Accounts" > Databricks-Notebook-Svc-Acct. 
# MAGIC 3. Select "Databricks-Notebook-Svc-Acct" > Add Key . This will save json on your local machine.  
# MAGIC 2. https://admin.google.com/
# MAGIC   a. Go to Security > API Controls > Settings. Enable "Trust internal, domain-owned apps"
# MAGIC   b. Click on "Manage Domain-wide Delegation". 
# MAGIC   c. Input the Client ID* (you can get this from the file downloaded in step 3 above) of your service account and in OAuth scopes input: https://www.googleapis.com/auth/gmail.readonly 
# MAGIC   

# COMMAND ----------

import base64
import imaplib
import json
from optparse import OptionParser
import smtplib
import sys
import urllib

# COMMAND ----------

# The URL root for accessing Google Accounts.
GOOGLE_ACCOUNTS_BASE_URL = 'https://accounts.google.com'


# Hardcoded dummy redirect URI for non-web apps.
# REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob'
REDIRECT_URI = 'https://e2-demo-field-eng.cloud.databricks.com'

# COMMAND ----------

my_email_address = "arun.wagle.bizai@gmail.com"
my_password = "sourceCode0520~"

# COMMAND ----------

def SetupOptionParser():
  # Usage message is the module's docstring.
  parser = OptionParser(usage=__doc__)
  parser.add_option('--generate_oauth2_token',
                    action='store_true',
                    dest='generate_oauth2_token',
                    help='generates an OAuth2 token for testing')
  parser.add_option('--generate_oauth2_string',
                    action='store_true',
                    dest='generate_oauth2_string',
                    help='generates an initial client response string for '
                         'OAuth2')
  parser.add_option('--client_id',
                    default=None,
                    help='Client ID of the application that is authenticating. '
                         'See OAuth2 documentation for details.')
  parser.add_option('--client_secret',
                    default=None,
                    help='Client secret of the application that is '
                         'authenticating. See OAuth2 documentation for '
                         'details.')
  parser.add_option('--access_token',
                    default=None,
                    help='OAuth2 access token')
  parser.add_option('--refresh_token',
                    default=None,
                    help='OAuth2 refresh token')
  parser.add_option('--scope',
                    default='https://mail.google.com/',
                    help='scope for the access token. Multiple scopes can be '
                         'listed separated by spaces with the whole argument '
                         'quoted.')
  parser.add_option('--test_imap_authentication',
                    action='store_true',
                    dest='test_imap_authentication',
                    help='attempts to authenticate to IMAP')
  parser.add_option('--test_smtp_authentication',
                    action='store_true',
                    dest='test_smtp_authentication',
                    help='attempts to authenticate to SMTP')
  parser.add_option('--user',
                    default=None,
                    help='email address of user whose account is being '
                         'accessed')
  parser.add_option('--quiet',
                    action='store_true',
                    default=False,
                    dest='quiet',
                    help='Omit verbose descriptions and only print '
                         'machine-readable outputs.')
  return parser

def AccountsUrl(command):
  """Generates the Google Accounts URL.

  Args:
    command: The command to execute.

  Returns:
    A URL for the given command.
  """
  return '%s/%s' % (GOOGLE_ACCOUNTS_BASE_URL, command)


def UrlEscape(text):
  # See OAUTH 5.1 for a definition of which characters need to be escaped.
  return urllib.parse.quote(text, safe='~-._')


def UrlUnescape(text):
  # See OAUTH 5.1 for a definition of which characters need to be escaped.
  return urllib.parse.unquote(text)


def FormatUrlParams(params):
  """Formats parameters into a URL query string.

  Args:
    params: A key-value map.

  Returns:
    A URL query string version of the given parameters.
  """
  param_fragments = []
  for param in sorted(params.items(), key=lambda x: x[0]):
    param_fragments.append('%s=%s' % (param[0], UrlEscape(param[1])))
  return '&'.join(param_fragments)


def GeneratePermissionUrl(client_id, scope='https://mail.google.com/'):
  """Generates the URL for authorizing access.

  This uses the "OAuth2 for Installed Applications" flow described at
  https://developers.google.com/accounts/docs/OAuth2InstalledApp

  Args:
    client_id: Client ID obtained by registering your app.
    scope: scope for access token, e.g. 'https://mail.google.com'
  Returns:
    A URL that the user should visit in their browser.
  """
  params = {}
  params['client_id'] = client_id
  params['redirect_uri'] = REDIRECT_URI
  params['scope'] = scope
  params['response_type'] = 'code'
  return '%s?%s' % (AccountsUrl('o/oauth2/v2/auth'),
                    FormatUrlParams(params))


def AuthorizeTokens(client_id, client_secret, authorization_code):
  """Obtains OAuth access token and refresh token.

  This uses the application portion of the "OAuth2 for Installed Applications"
  flow at https://developers.google.com/accounts/docs/OAuth2InstalledApp#handlingtheresponse

  Args:
    client_id: Client ID obtained by registering your app.
    client_secret: Client secret obtained by registering your app.
    authorization_code: code generated by Google Accounts after user grants
        permission.
  Returns:
    The decoded response from the Google Accounts server, as a dict. Expected
    fields include 'access_token', 'expires_in', and 'refresh_token'.
  """
  params = {}
  params['client_id'] = client_id
  params['client_secret'] = client_secret
  params['code'] = authorization_code
  params['redirect_uri'] = REDIRECT_URI
  params['grant_type'] = 'authorization_code'
  request_url = AccountsUrl('o/oauth2/token')

  response = urllib.urlopen(request_url, urllib.urlencode(params)).read()
  return json.loads(response)


def RefreshToken(client_id, client_secret, refresh_token):
  """Obtains a new token given a refresh token.

  See https://developers.google.com/accounts/docs/OAuth2InstalledApp#refresh

  Args:
    client_id: Client ID obtained by registering your app.
    client_secret: Client secret obtained by registering your app.
    refresh_token: A previously-obtained refresh token.
  Returns:
    The decoded response from the Google Accounts server, as a dict. Expected
    fields include 'access_token', 'expires_in', and 'refresh_token'.
  """
  params = {}
  params['client_id'] = client_id
  params['client_secret'] = client_secret
  params['refresh_token'] = refresh_token
  params['grant_type'] = 'refresh_token'
  request_url = AccountsUrl('o/oauth2/token')

  response = urllib.urlopen(request_url, urllib.urlencode(params)).read()
  return json.loads(response)


def GenerateOAuth2String(username, access_token, base64_encode=True):
  """Generates an IMAP OAuth2 authentication string.

  See https://developers.google.com/google-apps/gmail/oauth2_overview

  Args:
    username: the username (email address) of the account to authenticate
    access_token: An OAuth2 access token.
    base64_encode: Whether to base64-encode the output.

  Returns:
    The SASL argument for the OAuth2 mechanism.
  """
  auth_string = 'user=%s\1auth=Bearer %s\1\1' % (username, access_token)
  if base64_encode:
    auth_string = base64.b64encode(auth_string)
  return auth_string


def TestImapAuthentication(user, auth_string):
  """Authenticates to IMAP with the given auth_string.

  Prints a debug trace of the attempted IMAP connection.

  Args:
    user: The Gmail username (full email address)
    auth_string: A valid OAuth2 string, as returned by GenerateOAuth2String.
        Must not be base64-encoded, since imaplib does its own base64-encoding.
  """
  print
  imap_conn = imaplib.IMAP4_SSL('imap.gmail.com')
  imap_conn.debug = 4
  imap_conn.authenticate('XOAUTH2', lambda x: auth_string)
  imap_conn.select('INBOX')


def TestSmtpAuthentication(user, auth_string):
  """Authenticates to SMTP with the given auth_string.

  Args:
    user: The Gmail username (full email address)
    auth_string: A valid OAuth2 string, not base64-encoded, as returned by
        GenerateOAuth2String.
  """
  print
  smtp_conn = smtplib.SMTP('smtp.gmail.com', 587)
  smtp_conn.set_debuglevel(True)
  smtp_conn.ehlo('test')
  smtp_conn.starttls()
  smtp_conn.docmd('AUTH', 'XOAUTH2 ' + base64.b64encode(auth_string))
  
  
def RequireOptions(options, *args):
  missing = [arg for arg in args if getattr(options, arg) is None]
  if missing:
    print ('Missing options: %s' % ' '.join(missing))
    sys.exit(-1)
    
    
def main(argv):
  print ("######Calling main####::argv::{}".format(argv))
  options_parser = SetupOptionParser()
  (options, args) = options_parser.parse_args(argv)
  print ("######Calling main####::11::{}".format((options, args)))
  if options.refresh_token:
    RequireOptions(options, 'client_id', 'client_secret')
    response = RefreshToken(options.client_id, options.client_secret,
                            options.refresh_token)
    if options.quiet:
      print (response['access_token'])
    else:
      print ('Access Token: %s' % response['access_token'])
      print ('Access Token Expiration Seconds: %s' % response['expires_in'])
  elif options.generate_oauth2_string:
    RequireOptions(options, 'user', 'access_token')
    oauth2_string = GenerateOAuth2String(options.user, options.access_token)
    if options.quiet:
      print (oauth2_string)
    else:
      print ('OAuth2 argument:\n' + oauth2_string)
  elif options.generate_oauth2_token:
    RequireOptions(options, 'client_id', 'client_secret')
    print ('To authorize token, visit this url and follow the directions:')
    print ('  %s' % GeneratePermissionUrl(options.client_id, options.scope))
    authorization_code = raw_input('Enter verification code: ')
    response = AuthorizeTokens(options.client_id, options.client_secret,
                                authorization_code)
    print ('Refresh Token: %s' % response['refresh_token'])
    print ('Access Token: %s' % response['access_token'])
    print ('Access Token Expiration Seconds: %s' % response['expires_in'])
  elif options.test_imap_authentication:
    RequireOptions(options, 'user', 'access_token')
    TestImapAuthentication(options.user,
        GenerateOAuth2String(options.user, options.access_token,
                             base64_encode=False))
  elif options.test_smtp_authentication:
    RequireOptions(options, 'user', 'access_token')
    TestSmtpAuthentication(options.user,
        GenerateOAuth2String(options.user, options.access_token,
                             base64_encode=False))
  else:
    options_parser.print_help()
    print ('Nothing to do, exiting.')
    return

# COMMAND ----------

# --generate_oauth2_token --client_id=364545978226.apps.googleusercontent.com --client_secret=zNrNsBzOOnQy8_O-8LkofeTR
args = ["--generate_oauth2_token", "--client_id", "375944687977-mia9us957qpdgl71ca3g73uqmhsql3qg.apps.googleusercontent.com", "--client_secret", "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"]
#  "--scope", "https://www.googleapis.com/auth/gmail.readonly"
# scope=authorization_code

main(args)
