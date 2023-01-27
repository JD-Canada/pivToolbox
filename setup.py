from distutils.core import setup
setup(
  name = 'pivToolbox',         # How you named your package folder (MyLib)
  packages = ['pivToolbox'],   # Chose the same as "name"
  version = '0.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Growing collection of tools to process particle image velocimetry data',   # Give a short description about your library
  author = 'Jason Duguay',                   # Type in your name
  author_email = 'duguay.jason@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/Fishified/pivToolbox',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Fishified/pivToolbox/archive/0.1.zip',    # I explain this later on
  keywords = ['python', 'particle image velocimetry'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          '',
          '',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)