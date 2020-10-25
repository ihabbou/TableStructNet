

class Table: 
    def __init__(self):
        self.cells = []
    
    def __str__(self):
        return "table_id="+self.id+", \n"+\
            f"table_coords={self.coords}, \n"+\
            f"table_cells={len(self.cells)}: \n"+\
            "\n".join([str(cell) for cell in self.cells])+"\n"

class Cell:
    def __init__(self):
        pass
    def __str__(self):
        return "\tcell_id="+self.id+", \n"+\
            f"\tcell_coords={self.coords}, \n"+\
            f"\tcell_location={self.location} \n"

def parse_tables_from_xml(xml_path):
    """Parses the ICDAR2019 xml annotation into a list of objects.

    Parameters
    ----------
    xml_path : path to the ICDAR2019 xml annotation.

    Returns
    -------
    tables : a list of :ref:`Table` objects from the xml annotation.
    """
    
    from xml.dom import minidom
    xmldoc = minidom.parse(xml_path)
    
    table_objs = []

    tablelist = xmldoc.getElementsByTagName('table')

    for table in tablelist:
        table_obj = Table()

        table_id = table.attributes['id'].value
        table_obj.id = table_id

        coords = [node for node in table.childNodes 
                  if node.nodeType == table.ELEMENT_NODE][0].attributes["points"].value
        coords = tuple([tuple(map(int, point.split(','))) 
                        for point in coords.split()])
        table_obj.coords = coords

        for cell in table.getElementsByTagName("cell"):
            cell_obj = Cell()
            cell_id = cell.attributes['id'].value
            cell_obj.id = cell_id

            #start_row, end_row, start_col, end_col
            location = \
                cell.attributes["start-row"].value, \
                cell.attributes["end-row"].value, \
                cell.attributes["start-col"].value, \
                cell.attributes["end-col"].value
            location = tuple(map(int, location))
            cell_obj.location = location

            ccoords = [node for node in cell.childNodes 
                      if node.nodeType == cell.ELEMENT_NODE][0].attributes["points"].value
            ccoords = tuple([tuple(map(int, point.split(','))) 
                             for point in ccoords.split()])
            
            cell_obj.coords = ccoords

            table_obj.cells.append(cell_obj)
        table_objs.append(table_obj)
        
    return table_objs
