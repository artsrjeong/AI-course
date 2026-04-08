# streamlit_app.py

import streamlit as st
from sqlalchemy import text

# Create the SQL connection to pets_db as specified in your secrets file.
conn = st.connection('pets_db', type='sql')

# Insert some data with conn.session.
with conn.session as s:
    s.execute(text('CREATE TABLE IF NOT EXISTS pet_owners (person TEXT, pet TEXT);'))
    s.execute(text('DELETE FROM pet_owners;'))
    pet_owners = {'jerry': 'fish', 'barbara': 'cat', 'alex': 'puppy'}
    for k in pet_owners:
        s.execute(
            text('INSERT INTO pet_owners (person, pet) VALUES (:owner, :pet);'),
            params=dict(owner=k, pet=pet_owners[k])
        )
    s.commit()

# Query and display the data you inserted
pet_owners = conn.query('select * from pet_owners')
st.dataframe(pet_owners)