"""Physical Review article references, numbered in citation order."""

from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsortedStyle
from pybtex.style.template import field, href, join, sentence, tag, words


class APSStyle(UnsortedStyle):
    """Article titles link to the DOI; journal names use stored abbreviations."""

    default_name_style = 'plain'

    def __init__(self, **kwargs):
        super().__init__(abbreviate_names=True, **kwargs)

    def get_article_template(self, entry):
        title = field('title')
        if 'doi' in entry.fields:
            title = href[join['https://doi.org/', field('doi')], title]
        return sentence[join(sep=', ')[
            self.format_names('author', as_sentence=False), title,
            words[field('journal'), tag('strong')[field('volume')]],
            words[field('pages', apply_func=lambda value: str(value).split('–')[0].split('--')[0]),
                  join['(', field('year'), ')']],
        ]]


def setup(app):
    register_plugin('pybtex.style.formatting', 'aps', APSStyle)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
