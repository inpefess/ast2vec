# Scripts for serving a trained ast2vec model using TorchServe
# Copyright (C) 2022  Boris Shminke
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
How to use
==========

First, create a model archive file::

.. code:: sh

    torch-model-archiver --model-name ast2vec --version 1.0\
        --model-file ast2vec.py --serialized-file ast2vec.pt\
        --export-path model_store\
        --handler handler:entry_point_function_name

Then, start serving the model archive::

.. code:: sh

    torchserve --start --ncs --model-store model_store\
         --models ast2vec.mar

See ``config.properties`` to set the ports and parallelism.

Finally, use REST API to get encodings::

.. code:: sh

    curl http://127.0.0.1:9080/predictions/ast2vec\
         -H 'Content-Type: application/json'\
         -d '{"data": "print(\"Hello, world\")"}'

"""
import ast2vec
import ast
import python_ast_utils
from hashlib import sha256
from pymemcache.client import Client
import json

model = None
memcached = Client("memcached")

def entry_point_function_name(data, context):
    global model, memcached

    if not data:
        model = ast2vec.load_model()
    else:
        raw_key = data[0]["body"]["data"]
        hashed_key = sha256(raw_key.encode("utf8")).hexdigest()
        result = memcached.get(hashed_key)
        if result:
            return json.loads(result.decode("utf8"))
        tree = python_ast_utils.ast_to_tree(ast.parse(raw_key))
        result = [model.encode(tree).detach().numpy().tolist()]
        memcached.set(hashed_key, result)
        return result
