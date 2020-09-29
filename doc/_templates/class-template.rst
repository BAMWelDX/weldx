{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__init__',
                                                  '__eq__',
                                                  '__repr__',
                                                  '__str__',
                                                  '__add__',
                                                  '__sub__',
                                                  ] %}
       ~{{ name }}.{{ item }}
       {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

