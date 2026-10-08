//! The object records of one crate image.

mod abi;

use polyasm::object::InterchangeFragment;

use super::model::PortableImage;

impl PortableImage {
    /// Answers the interchange records the crate's object carries.
    pub(crate) fn relocatable(&self) -> InterchangeFragment {
        self.fragment.clone()
    }
}
