"""Order-preserving many-to-one target item lookup."""
import pyarrow as pa
import pyarrow.compute as pc


def attach_items(exposures, items):
    if items['205'].null_count or pc.count_distinct(items['205']).as_py() != len(items):
        raise ValueError('Item keys must be non-null and unique')
    indices = pc.index_in(exposures['205'], value_set=items['205'])
    if indices.null_count:
        raise ValueError(f'Missing target attributes for {indices.null_count} exposures')
    result = exposures
    for name in ['206', '213', '214']:
        values = pc.take(items[name], indices)
        if values.null_count:
            raise ValueError('Null target attributes: ' + name)
        result = result.append_column(name, values)
    return result
