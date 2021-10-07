import pandas as pd


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)


def round_value(value, binary=False):
    divisor = 1024. if binary else 1000.

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + 'T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + 'G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + 'M'
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + 'K'
    return str(value)


def report_format(collected_nodes):
    data = list()
    for node in collected_nodes:
        name = node.name
        Flops = node.Flops
        data.append([name, Flops])
    df = pd.DataFrame(data)
    df.columns = ['module name', 'Flops']
    total_flops = df['Flops'].sum()

    # Add Total row
    total_df = pd.Series([total_flops
                         ],
                         index=['Flops'],
                         name='total')
    df = df.append(total_df)

    df = df.fillna(' ')

    df['Flops'] = df['Flops'].apply(lambda x: '{:,}'.format(x))

    summary = str(df) + '\n'
    summary += "=" * len(str(df).split('\n')[0])
    summary += '\n'

    summary += "-" * len(str(df).split('\n')[0])
    summary += '\n'

    summary += "Total Flops: {}Flops\n".format(round_value(total_flops))

    return summary
