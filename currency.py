#coding: utf-8
import re
from babel.numbers import get_currency_name, get_currency_symbol

# The list of currency codes copied from "http://www.xe.com/iso4217.php"
CURRENCY_CODES = [
  ("AED", "United Arab Emirates Dirham"),
  ("AFN", "Afghanistan Afghani"),
  ("ALL", "Albania Lek"),
  ("AMD", "Armenia Dram"),
  ("ANG", "Netherlands Antilles Guilder"),
  ("AOA", "Angola Kwanza"),
  ("ARS", "Argentina Peso"),
  ("AUD", "Australia Dollar"),
  ("AWG", "Aruba Guilder"),
  ("AZN", "Azerbaijan Manat"),
  ("BAM", "Bosnia and Herzegovina Convertible Marka"),
  ("BBD", "Barbados Dollar"),
  ("BDT", "Bangladesh Taka"),
  ("BGN", "Bulgaria Lev"),
  ("BHD", "Bahrain Dinar"),
  ("BIF", "Burundi Franc"),
  ("BMD", "Bermuda Dollar"),
  ("BND", "Brunei Darussalam Dollar"),
  ("BOB", "Bolivia Bolíviano"),
  ("BRL", "Brazil Real"),
  ("BSD", "Bahamas Dollar"),
  ("BTN", "Bhutan Ngultrum"),
  ("BWP", "Botswana Pula"),
  ("BYN", "Belarus Ruble"),
  ("BZD", "Belize Dollar"),
  ("CAD", "Canada Dollar"),
  ("CDF", "Congo/Kinshasa Franc"),
  ("CHF", "Switzerland Franc"),
  ("CLP", "Chile Peso"),
  ("CNY", "China Yuan Renminbi"),
  ("COP", "Colombia Peso"),
  ("CRC", "Costa Rica Colon"),
  ("CUC", "Cuba Convertible Peso"),
  ("CUP", "Cuba Peso"),
  ("CVE", "Cape Verde Escudo"),
  ("CZK", "Czech Republic Koruna"),
  ("DJF", "Djibouti Franc"),
  ("DKK", "Denmark Krone"),
  ("DOP", "Dominican Republic Peso"),
  ("DZD", "Algeria Dinar"),
  ("EGP", "Egypt Pound"),
  ("ERN", "Eritrea Nakfa"),
  ("ETB", "Ethiopia Birr"),
  ("EUR", "Euro Member Countries"),
  ("FJD", "Fiji Dollar"),
  ("FKP", "Falkland Islands (Malvinas) Pound"),
  ("GBP", "United Kingdom Pound"),
  ("GEL", "Georgia Lari"),
  ("GGP", "Guernsey Pound"),
  ("GHS", "Ghana Cedi"),
  ("GIP", "Gibraltar Pound"),
  ("GMD", "Gambia Dalasi"),
  ("GNF", "Guinea Franc"),
  ("GTQ", "Guatemala Quetzal"),
  ("GYD", "Guyana Dollar"),
  ("HKD", "Hong Kong Dollar"),
  ("HNL", "Honduras Lempira"),
  ("HRK", "Croatia Kuna"),
  ("HTG", "Haiti Gourde"),
  ("HUF", "Hungary Forint"),
  ("IDR", "Indonesia Rupiah"),
  ("ILS", "Israel Shekel"),
  ("IMP", "Isle of Man Pound"),
  ("INR", "India Rupee"),
  ("IQD", "Iraq Dinar"),
  ("IRR", "Iran Rial"),
  ("ISK", "Iceland Krona"),
  ("JEP", "Jersey Pound"),
  ("JMD", "Jamaica Dollar"),
  ("JOD", "Jordan Dinar"),
  ("JPY", "Japan Yen"),
  ("KES", "Kenya Shilling"),
  ("KGS", "Kyrgyzstan Som"),
  ("KHR", "Cambodia Riel"),
  ("KMF", "Comorian Franc"),
  ("KPW", "Korea (North) Won"),
  ("KRW", "Korea (South) Won"),
  ("KWD", "Kuwait Dinar"),
  ("KYD", "Cayman Islands Dollar"),
  ("KZT", "Kazakhstan Tenge"),
  ("LAK", "Laos Kip"),
  ("LBP", "Lebanon Pound"),
  ("LKR", "Sri Lanka Rupee"),
  ("LRD", "Liberia Dollar"),
  ("LSL", "Lesotho Loti"),
  ("LYD", "Libya Dinar"),
  ("MAD", "Morocco Dirham"),
  ("MDL", "Moldova Leu"),
  ("MGA", "Madagascar Ariary"),
  ("MKD", "Macedonia Denar"),
  ("MMK", "Myanmar (Burma) Kyat"),
  ("MNT", "Mongolia Tughrik"),
  ("MOP", "Macau Pataca"),
  ("MRO", "Mauritania Ouguiya"),
  ("MUR", "Mauritius Rupee"),
  ("MVR", "Maldives (Maldive Islands) Rufiyaa"),
  ("MWK", "Malawi Kwacha"),
  ("MXN", "Mexico Peso"),
  ("MYR", "Malaysia Ringgit"),
  ("MZN", "Mozambique Metical"),
  ("NAD", "Namibia Dollar"),
  ("NGN", "Nigeria Naira"),
  ("NIO", "Nicaragua Cordoba"),
  ("NOK", "Norway Krone"),
  ("NPR", "Nepal Rupee"),
  ("NZD", "New Zealand Dollar"),
  ("OMR", "Oman Rial"),
  ("PAB", "Panama Balboa"),
  ("PEN", "Peru Sol"),
  ("PGK", "Papua New Guinea Kina"),
  ("PHP", "Philippines Peso"),
  ("PKR", "Pakistan Rupee"),
  ("PLN", "Poland Zloty"),
  ("PYG", "Paraguay Guarani"),
  ("QAR", "Qatar Riyal"),
  ("RON", "Romania Leu"),
  ("RSD", "Serbia Dinar"),
  ("RUB", "Russia Ruble"),
  ("RWF", "Rwanda Franc"),
  ("SAR", "Saudi Arabia Riyal"),
  ("SBD", "Solomon Islands Dollar"),
  ("SCR", "Seychelles Rupee"),
  ("SDG", "Sudan Pound"),
  ("SEK", "Sweden Krona"),
  ("SGD", "Singapore Dollar"),
  ("SHP", "Saint Helena Pound"),
  ("SLL", "Sierra Leone Leone"),
  ("SOS", "Somalia Shilling"),
  #("SPL*", "Seborga Luigino"),
  ("SPL", "Seborga Luigino"),
  ("SRD", "Suriname Dollar"),
  ("STD", "São Tomé and Príncipe Dobra"),
  ("SVC", "El Salvador Colon"),
  ("SYP", "Syria Pound"),
  ("SZL", "Swaziland Lilangeni"),
  ("THB", "Thailand Baht"),
  ("TJS", "Tajikistan Somoni"),
  ("TMT", "Turkmenistan Manat"),
  ("TND", "Tunisia Dinar"),
  ("TOP", "Tonga Pa'anga"),
  ("TRY", "Turkey Lira"),
  ("TTD", "Trinidad and Tobago Dollar"),
  ("TVD", "Tuvalu Dollar"),
  ("TWD", "Taiwan New Dollar"),
  ("TZS", "Tanzania Shilling"),
  ("UAH", "Ukraine Hryvnia"),
  ("UGX", "Uganda Shilling"),
  ("USD", "United States Dollar"),
  ("UYU", "Uruguay Peso"),
  ("UZS", "Uzbekistan Som"),
  ("VEF", "Venezuela Bolívar"),
  ("VND", "Viet Nam Dong"),
  ("VUV", "Vanuatu Vatu"),
  ("WST", "Samoa Tala"),
  ("XAF", "Communauté Financière Africaine (BEAC) CFA Franc BEAC"),
  ("XCD", "East Caribbean Dollar"),
  ("XDR", "International Monetary Fund (IMF) Special Drawing Rights"),
  ("XOF", "Communauté Financière Africaine (BCEAO) Franc"),
  ("XPF", "Comptoirs Français du Pacifique (CFP) Franc"),
  ("YER", "Yemen Rial"),
  ("ZAR", "South Africa Rand"),
  ("ZMW", "Zambia Kwacha"),
  ("ZWD", "Zimbabwe Dollar"),
]


CURRENCY_SYMBOLS = [get_currency_symbol(c[0], locale="en.US") for c in CURRENCY_CODES]
CURRENCY_NAMES = [get_currency_name(c[0], locale="en.US") for c in CURRENCY_CODES]


def get_currency_tokens(use_currency_name=True):
  currency_symbols = [t for t in CURRENCY_SYMBOLS if not re.search("[A-Za-z]", t)] # '$, ₣, ...'

  # Pick up only the units of currency. (dollar, franc, ...)
  currency_names = []
  if use_currency_name:
    # List up major currency names for now.
    #currency_names = ['Dollar', 'Euro', 'Yen', 'Franc', 'Pound', 'Won']
    currency_names = [n.split()[-1].lower() for n in CURRENCY_NAMES if not re.search("[0-9\(\)]", n)]

  # TODO: if they are lemmatized and converted to lower case, irrelevant words can be contained (e.g. 'all', 'imp', 'rand')
  currency_symbols = list(set(currency_symbols))
  currency_names = list(set(currency_names))
  removal_names = ['real', 'rights', 'mark', 'won'] # Currency names with the same spelling as common words are removed.
  
  for c in removal_names:
    currency_names.remove(c)

  plurals = [c + 's' for c in currency_names] # I don't know whether all currency units have a plural form....
  currency_names += plurals
  return set(currency_symbols), set(currency_names)

if __name__ == '__main__':
  symbols, names = get_currency_tokens()
  print ', '.join(symbols)
  print ', '.join(names)

  #₡, £, ¥, $, ₦, ₩, ₫, ₪, ₭, €, ₮, ₱, ₲, ₴, ₹, ₸, ₺, ₽, ฿, ₾, ៛, ৳
  #ariary, birr, yen, dollar, lilangeni, byn, imp, hryvnia, dalasi, lira, paʻanga, real, koruna, kwanza, sol, rufiyaa, ouguiya, rights, manat, naira, vatu, zloty, tvd, riel, kwacha, ringgit, kyat, cedi, loti, won, afghani, lari, balboa, tugrik, franc, ggp, forint, baht, lek, leu, lev, metical, kuna, bolívar, rand, som, denar, króna, escudo, dong, jep, nakfa, shilling, mark, dirham, sheqel, krona, krone, ruble, somoni, córdoba, ngultrum, kina, boliviano, pula, riyal, peso, pataca, pound, spl, tenge, florin, gourde, colón, taka, rupiah, rial, kip, dram, yuan, euro, quetzal, guarani, guilder, rupee, dinar, lempira, tala, dobra, leone
